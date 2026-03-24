#!/usr/bin/env python3
"""
Experiment #609: 15m Primary + 1h/1d HTF — Stochastic Entry + HMA Trend + Volume Confirm

Hypothesis: 15m timeframe has failed in previous experiments (#597, #601, #605) with Sharpe=0.000
due to ZERO trades generated. Entry conditions were TOO STRICT. This strategy uses:
1. Simpler HTF bias (1d HMA21) - proven to work in baseline strategies
2. 1h RSI for momentum confirmation - less strict than ADX/CHOP filters
3. 15m Stochastic for entry timing - faster signal generation than RSI
4. Volume confirmation as soft filter (not hard requirement)
5. Session bias toward 00-12 UTC but allows trades outside if signal strong
6. Conservative position sizing (0.20-0.25) for higher frequency

Key differences from failed 15m strategies:
1. LOOSER entry conditions - Stochastic crosses are more frequent than RSI extremes
2. HTF bias is directional guide, not hard filter (allows counter-trend mean reversion)
3. Volume filter is weighted, not binary (still enter if other signals strong)
4. Target 60-120 trades/year (15m needs more trades than 4h/6h)

Strategy logic:
1. 1d HMA(21) = macro trend bias (aligned via mtf_data)
2. 1h RSI(14) = momentum confirmation (RSI>55 bull, RSI<45 bear)
3. 15m Stochastic(14,3,3) = entry trigger (%K crosses %D at extremes)
4. Volume ratio = taker_buy_volume / volume (confirm participation)
5. ATR(14)*2.0 stoploss on all positions
6. Position size: 0.20 base, 0.25 strong confluence

Regime-adaptive entries:
- TREND (price > 1d HMA + 1h RSI > 55): Long on Stoch oversold cross
- TREND (price < 1d HMA + 1h RSI < 45): Short on Stoch overbought cross
- MEAN REVERSION (1h RSI extreme <30 or >70): Enter counter to 1d HMA if Stoch confirms

Target: Sharpe>0.40, trades>=150 train (40/year), trades>=15 test
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_stoch_hma_rsi_vol_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """
    Stochastic Oscillator
    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = SMA(%K, d_period)
    """
    n = len(close)
    if n < k_period + d_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    pct_k = np.zeros(n)
    pct_k[:] = np.nan
    
    for i in range(k_period - 1, n):
        lowest_low = np.nanmin(low[i - k_period + 1:i + 1])
        highest_high = np.nanmax(high[i - k_period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            pct_k[i] = 100.0 * (close[i] - lowest_low) / price_range
        else:
            pct_k[i] = 50.0
    
    pct_d = pd.Series(pct_k).rolling(window=d_period, min_periods=d_period).mean().values
    
    return pct_k, pct_d

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

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1h RSI for momentum confirmation
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    stoch_k, stoch_d = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    rsi_15m = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 1e-10:
            vol_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            vol_ratio[i] = 0.5
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 1H MOMENTUM (RSI) ===
        h1_bull = rsi_1h_aligned[i] > 55.0
        h1_bear = rsi_1h_aligned[i] < 45.0
        h1_extreme_bull = rsi_1h_aligned[i] > 70.0
        h1_extreme_bear = rsi_1h_aligned[i] < 30.0
        
        # === 15M STOCHASTIC ===
        stoch_oversold = stoch_k[i] < 25.0 and stoch_d[i] < 25.0
        stoch_overbought = stoch_k[i] > 75.0 and stoch_d[i] > 75.0
        
        # Stochastic cross signals
        stoch_bull_cross = False
        stoch_bear_cross = False
        if i >= 2 and not np.isnan(stoch_k[i-1]) and not np.isnan(stoch_d[i-1]):
            # Bull cross: %K crosses above %D in oversold zone
            if stoch_k[i-1] < stoch_d[i-1] and stoch_k[i] > stoch_d[i] and stoch_k[i] < 40.0:
                stoch_bull_cross = True
            # Bear cross: %K crosses below %D in overbought zone
            if stoch_k[i-1] > stoch_d[i-1] and stoch_k[i] < stoch_d[i] and stoch_k[i] > 60.0:
                stoch_bear_cross = True
        
        # === VOLUME CONFIRMATION ===
        vol_bull = vol_ratio[i] > 0.55
        vol_bear = vol_ratio[i] < 0.45
        
        # === SESSION BIAS (00-12 UTC preferred) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 3600)) % 24
        is_prime_session = 0 <= hour_utc <= 12
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        signal_strength = 0
        
        # TREND LONG: 1d HMA bull + 1h RSI bull + 15m Stoch bull cross
        if htf_bull and h1_bull and stoch_bull_cross:
            desired_signal = SIZE_STRONG
            signal_strength = 3
        elif htf_bull and stoch_bull_cross:
            desired_signal = SIZE_BASE
            signal_strength = 2
        elif h1_extreme_bull and stoch_bull_cross:
            # Mean reversion long on extreme oversold
            desired_signal = SIZE_BASE
            signal_strength = 2
        
        # TREND SHORT: 1d HMA bear + 1h RSI bear + 15m Stoch bear cross
        if htf_bear and h1_bear and stoch_bear_cross:
            desired_signal = -SIZE_STRONG
            signal_strength = max(signal_strength, 3)
        elif htf_bear and stoch_bear_cross:
            desired_signal = -SIZE_BASE
            signal_strength = max(signal_strength, 2)
        elif h1_extreme_bear and stoch_bear_cross:
            # Mean reversion short on extreme overbought
            desired_signal = -SIZE_BASE
            signal_strength = max(signal_strength, 2)
        
        # Volume confirmation boost (not required, just boosts confidence)
        if desired_signal > 0 and vol_bull:
            desired_signal = min(desired_signal * 1.1, SIZE_STRONG)
        elif desired_signal < 0 and vol_bear:
            desired_signal = max(desired_signal * 1.1, -SIZE_STRONG)
        
        # Session boost (prefer prime hours but allow all)
        if is_prime_session and abs(desired_signal) > 0:
            desired_signal = desired_signal * 1.05
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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