#!/usr/bin/env python3
"""
Experiment #241: 15m Bollinger Squeeze Breakout with 1h/4h HMA Trend Filter
Hypothesis: Bollinger Band squeeze (low volatility) followed by breakout with volume 
confirmation captures momentum moves early. 1h HMA provides primary trend bias, 4h HMA 
confirms macro direction. KAMA adapts to market conditions better than EMA for entries.
RSI filter ensures we're not entering at extremes. Position sizing: 0.25 entry, 0.125 
half at 2R profit. Stoploss: 2.0*ATR trailing stop. Target: Beat Sharpe=0.499.
Key difference from failures: Looser BB squeeze threshold (1.0 instead of 0.5) and 
RSI range (25-75) to ensure sufficient trades on all symbols.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_bb_squeeze_1h_4h_hma_kama_volume_atr_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close_s - close_s.shift(period))
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    er = change / volatility
    er = er.fillna(0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[period-1] = close_s.iloc[period-1]
    for i in range(period, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    return kama.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma  # Bandwidth
    return upper.values, lower.values, sma.values, bw.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    return ratio

def calculate_volume_spike(volume, period=20):
    """Detect volume spike (>1.5x average)."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    spike = volume > 1.5 * vol_avg.values
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, 10)
    bb_upper, bb_lower, bb_mid, bb_bw = calculate_bollinger(close, 20, 2.0)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    vol_spike = calculate_volume_spike(volume, 20)
    
    # Track previous values for breakout detection
    prev_bb_bw = np.roll(bb_bw, 1)
    prev_bb_bw[0] = bb_bw[0]
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters (looser to ensure trades)
        hourly_bullish = close[i] > hma_1h_aligned[i]
        hourly_bearish = close[i] < hma_1h_aligned[i]
        fourh_bullish = close[i] > hma_4h_aligned[i]
        fourh_bearish = close[i] < hma_4h_aligned[i]
        
        # Bollinger Band squeeze detection (low volatility before breakout)
        bb_squeeze = bb_bw[i] < 0.10  # Bandwidth < 10% = squeeze
        bb_expansion = bb_bw[i] > prev_bb_bw[i] and bb_bw[i] > 0.08  # Expanding
        
        # Price position relative to bands
        near_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        near_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        above_mid = close[i] > bb_mid[i]
        below_mid = close[i] < bb_mid[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i] and kama[i] > prev_kama[i]
        kama_bearish = close[i] < kama[i] and kama[i] < prev_kama[i]
        kama_cross_up = prev_close[i] < prev_kama[i] and close[i] > kama[i]
        kama_cross_down = prev_close[i] > prev_kama[i] and close[i] < kama[i]
        
        # RSI filter (looser: 25-75 range to ensure trades)
        rsi_not_extreme = 25 < rsi[i] < 75
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        vol_confirmed = vol_spike[i]
        
        # Price momentum
        price_momentum = (close[i] - prev_close[i]) / prev_close[i]
        momentum_bullish = price_momentum > 0.002  # >0.2% gain
        momentum_bearish = price_momentum < -0.002  # >0.2% loss
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # BB squeeze breakout long with trend
        if bb_squeeze and bb_expansion and momentum_bullish:
            if hourly_bullish and kama_bullish and rsi_bullish:
                new_signal = SIZE_ENTRY
            elif fourh_bullish and vol_confirmed and rsi_not_extreme:
                new_signal = SIZE_ENTRY
        
        # KAMA cross up with trend confirmation
        elif kama_cross_up:
            if hourly_bullish and above_mid and rsi_not_extreme:
                new_signal = SIZE_ENTRY
            elif fourh_bullish and vol_bullish and rsi_bullish:
                new_signal = SIZE_ENTRY
        
        # Pullback to BB mid in uptrend
        elif above_mid and hourly_bullish:
            if prev_close[i] < bb_mid[i] and close[i] > bb_mid[i]:
                if kama_bullish or vol_bullish:
                    new_signal = SIZE_ENTRY
        
        # RSI oversold bounce in uptrend
        elif rsi_oversold and hourly_bullish:
            if close[i] > prev_close[i] and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # BB squeeze breakout short with trend
        if bb_squeeze and bb_expansion and momentum_bearish:
            if hourly_bearish and kama_bearish and rsi_bearish:
                new_signal = -SIZE_ENTRY
            elif fourh_bearish and vol_confirmed and rsi_not_extreme:
                new_signal = -SIZE_ENTRY
        
        # KAMA cross down with trend confirmation
        elif kama_cross_down:
            if hourly_bearish and below_mid and rsi_not_extreme:
                new_signal = -SIZE_ENTRY
            elif fourh_bearish and vol_bearish and rsi_bearish:
                new_signal = -SIZE_ENTRY
        
        # Pullback to BB mid in downtrend
        elif below_mid and hourly_bearish:
            if prev_close[i] > bb_mid[i] and close[i] < bb_mid[i]:
                if kama_bearish or vol_bearish:
                    new_signal = -SIZE_ENTRY
        
        # RSI overbought rejection in downtrend
        elif rsi_overbought and hourly_bearish:
            if close[i] < prev_close[i] and vol_bearish:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals