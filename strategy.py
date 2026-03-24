#!/usr/bin/env python3
"""
Experiment #957: 15m Primary + 4h/12h HTF — RSI Pullback + Volume Confirmation

Hypothesis: 15m timeframe with HTF trend filter + RSI(7) pullback entries + volume
confirmation will generate sufficient trades while maintaining positive Sharpe.

Key innovations:
1. 4h HMA(21) for primary trend direction (call ONCE before loop)
2. 12h momentum (close vs SMA50) for regime filter (call ONCE before loop)
3. 15m RSI(7) for entry timing: oversold <30, overbought >70 (LOOSE thresholds)
4. Volume confirmation: taker_buy_ratio > 0.52 for long, <0.48 for short
5. Session bias: UTC 00-12 preferred (London/NY overlap) but not required
6. ATR(14) 2.5x trailing stop for risk management

Why this should work on 15m:
- HTF filters prevent counter-trend trades (major failure mode)
- RSI(7) is faster than RSI(14), catches intraday reversals
- Volume confirmation filters false breakouts
- LOOSE thresholds ensure trades generate (learning from 0-trade failures)
- Smaller position size (0.15-0.25) for higher frequency TF

Target: Sharpe>0.3, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_vol_pullback_4h12h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan, dtype=np.float64)
    cumsum = np.cumsum(close)
    for i in range(period - 1, n):
        sma[i] = (cumsum[i] - cumsum[i - period] + close[i - period]) / period
    
    return sma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    sma_12h_raw = calculate_sma(df_12h['close'].values, period=50)
    sma_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_12h_raw)
    
    # 12h momentum: close vs SMA50
    mom_12h_raw = (df_12h['close'].values - sma_12h_raw) / (sma_12h_raw + 1e-10)
    mom_12h_aligned = align_htf_to_ltf(prices, df_12h, mom_12h_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Taker buy ratio (volume confirmation)
    taker_ratio = np.divide(taker_buy_vol, volume, out=np.zeros_like(volume), where=volume > 0)
    
    # Session filter: UTC hour from open_time (milliseconds)
    utc_hours = np.array([(t // 3600000) % 24 for t in open_time], dtype=np.int32)
    is_preferred_session = (utc_hours >= 0) & (utc_hours <= 12)
    
    signals = np.zeros(n)
    
    # Position sizing for 15m (smaller due to higher frequency)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    SIZE_MAX = 0.25
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(mom_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h HMA + 12h momentum) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_12h_bull = mom_12h_aligned[i] > 0.02  # >2% above SMA50
        htf_12h_bear = mom_12h_aligned[i] < -0.02  # <2% below SMA50
        
        # === VOLUME CONFIRMATION ===
        vol_bull = taker_ratio[i] > 0.52
        vol_bear = taker_ratio[i] < 0.48
        
        # === RSI EXTREMES (LOOSE THRESHOLDS FOR TRADES) ===
        rsi_oversold = rsi_7[i] < 30  # Very loose for 15m
        rsi_overbought = rsi_7[i] > 70
        
        rsi_oversold_strong = rsi_7[i] < 25
        rsi_overbought_strong = rsi_7[i] > 75
        
        # === SESSION BIAS ===
        session_boost = is_preferred_session[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG entries: HTF bull + RSI oversold + volume confirmation
        if htf_4h_bull or htf_12h_bull:  # Either HTF bullish
            if rsi_oversold and vol_bull:
                if rsi_oversold_strong and session_boost:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif rsi_7[i] < 40 and htf_4h_bull and htf_12h_bull and vol_bull:
                # Strong confluence even without extreme RSI
                desired_signal = SIZE_BASE
        
        # SHORT entries: HTF bear + RSI overbought + volume confirmation
        elif htf_4h_bear or htf_12h_bear:  # Either HTF bearish
            if rsi_overbought and vol_bear:
                if rsi_overbought_strong and session_boost:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif rsi_7[i] > 60 and htf_4h_bear and htf_12h_bear and vol_bear:
                # Strong confluence even without extreme RSI
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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