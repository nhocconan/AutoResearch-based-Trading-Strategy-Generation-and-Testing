#!/usr/bin/env python3
"""
Experiment #625: 15m Primary + 4h/1d HTF — Daily Pivot Mean Reversion + HTF Trend

Hypothesis: 15m timeframe can work if we use HTF for DIRECTION and 15m only for ENTRY TIMING.
Key insight from failed 15m experiments: too many filters = 0 trades. Need LOOSE entries with HTF bias.

Strategy logic:
1. 1d HMA(21) = macro trend bias (long only above, short only below)
2. 4h HMA(21) = medium trend confirmation (boosts signal size)
3. 15m RSI(7) = fast mean reversion entry (oversold <25 in uptrend, overbought >75 in downtrend)
4. 15m Choppiness(14) = regime filter (>60 = skip entries, avoid chop)
5. 15m ATR(14) = stoploss (2.0*ATR trailing)
6. Session filter: prefer 00-12 UTC (London+NY overlap)

Why this should work:
- HTF trend filter ensures we trade with the big picture
- Fast RSI(7) generates entries on pullbacks (not waiting for RSI 14)
- Choppiness filter avoids whipsaw in ranging markets
- Session filter reduces low-volume noise
- Small position size (0.15-0.20) for 15m frequency

Target: Sharpe>0.40, trades>=40 train, trades>=5 test, DD>-30%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher TF frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_daily_pivot_rsi_hma_4h1d_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_fast = calculate_rsi(close, period=7)  # Fast RSI for entries
    rsi_std = calculate_rsi(close, period=14)  # Standard RSI for confirmation
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_fast[i]) or np.isnan(chop[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        utc_hour = get_utc_hour(open_time[i])
        is_peak_session = 0 <= utc_hour <= 12
        
        # === HTF BIAS (1d primary, 4h confirmation) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # 4h medium-term confirmation
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 60.0
        
        # === RSI MEAN REVERSION ZONES (FAST RSI for entries) ===
        # Long: RSI(7) oversold in uptrend
        # Short: RSI(7) overbought in downtrend
        rsi_os_long = rsi_fast[i] < 30.0  # Oversold
        rsi_ob_short = rsi_fast[i] > 70.0  # Overbought
        
        # RSI momentum confirmation
        rsi_momentum_long = i > 1 and rsi_fast[i] > rsi_fast[i-1] if not np.isnan(rsi_fast[i-1]) else False
        rsi_momentum_short = i > 1 and rsi_fast[i] < rsi_fast[i-1] if not np.isnan(rsi_fast[i-1]) else False
        
        # === LOCAL TREND ===
        local_bull = close[i] > hma_15m[i]
        local_bear = close[i] < hma_15m[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS to ensure trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bull + RSI oversold + optional confirmations
        if htf_bull and rsi_os_long:
            # Strong signal: 4h also bull + session peak + RSI momentum
            if htf_4h_bull and is_peak_session and rsi_momentum_long:
                if is_choppy:
                    desired_signal = SIZE_BASE  # Reduce in chop
                else:
                    desired_signal = SIZE_STRONG
            # Base signal: 1d bull + RSI oversold (minimum)
            elif local_bull or rsi_momentum_long:
                if is_choppy:
                    desired_signal = SIZE_BASE * 0.5
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 1d bear + RSI overbought + optional confirmations
        elif htf_bear and rsi_ob_short:
            # Strong signal: 4h also bear + session peak + RSI momentum
            if htf_4h_bear and is_peak_session and rsi_momentum_short:
                if is_choppy:
                    desired_signal = -SIZE_BASE  # Reduce in chop
                else:
                    desired_signal = -SIZE_STRONG
            # Base signal: 1d bear + RSI overbought (minimum)
            elif local_bear or rsi_momentum_short:
                if is_choppy:
                    desired_signal = -SIZE_BASE * 0.5
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals