#!/usr/bin/env python3
"""
Experiment #007: 15m Bollinger Mean Reversion + 4h HMA Trend Bias + Volume Confirmation + ATR Stop
Hypothesis: 15m timeframe captures short-term mean reversion opportunities while 4h HMA provides
trend bias to avoid counter-trend trades. Bollinger Band width detects regime (squeeze=range,
expansion=trend). Volume confirmation filters false breakouts. Multiple entry paths ensure
>=10 trades per symbol. Conservative sizing (0.25) with 2.0*ATR stoploss controls drawdown.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_bb_meanrev_4h_hma_vol_atr_v2"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, sma, bandwidth

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def calculate_bandwidth_percentile(bandwidth, lookback=100):
    """Calculate percentile rank of bandwidth over lookback period."""
    n = len(bandwidth)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(bandwidth[i]):
            window = bandwidth[i-lookback+1:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                percentile[i] = np.sum(valid < bandwidth[i]) / len(valid) * 100
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    bw_percentile = calculate_bandwidth_percentile(bb_bandwidth, 100)
    
    # 15m HMA for additional trend confirmation
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bw_percentile[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m HMA trend
    ltf_bullish = close[i] > hma_15m[i]
        ltf_bearish = close[i] < hma_15m[i]
        hma_rising = hma_15m[i] > hma_15m[i-1] if i > 0 else False
        hma_falling = hma_15m[i] < hma_15m[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_15m_fast[i] > hma_15m[i]
        fast_below_slow = hma_15m_fast[i] < hma_15m[i]
        
        # Bollinger Band position
        price_vs_bb_upper = close[i] >= bb_upper[i]
        price_vs_bb_lower = close[i] <= bb_lower[i]
        price_vs_bb_mid = close[i] > bb_sma[i]
        
        # Bandwidth regime (squeeze vs expansion)
        regime_range = bw_percentile[i] < 40  # Low bandwidth = ranging
        regime_trend = bw_percentile[i] > 60  # High bandwidth = trending
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 20
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 60
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        vol_normal = volume[i] <= 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else True
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Mean reversion - price at BB lower + RSI oversold + HTF not bearish
        if price_vs_bb_lower and rsi_oversold and not htf_bearish:
            new_signal = SIZE_ENTRY
        
        # Path 2: Trend pullback - HTF bullish + price near BB mid + RSI pullback + ADX strong
        elif htf_bullish and not price_vs_bb_lower and rsi_pullback_long and trend_strong:
            new_signal = SIZE_ENTRY
        
        # Path 3: HMA crossover + HTF bullish + volume confirmation
        elif htf_bullish and fast_above_slow and hma_rising and vol_spike:
            new_signal = SIZE_ENTRY
        
        # Path 4: Range regime mean reversion - BB lower + RSI oversold + ADX weak
        elif regime_range and price_vs_bb_lower and rsi_oversold and trend_weak:
            new_signal = SIZE_ENTRY
        
        # Path 5: BB bounce - price crosses above BB lower from below + RSI rising
        elif price_vs_bb_lower and i > 0 and close[i-1] < bb_lower[i-1] and rsi[i] > rsi[i-1]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Mean reversion - price at BB upper + RSI overbought + HTF not bullish
        if price_vs_bb_upper and rsi_overbought and not htf_bullish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Trend pullback - HTF bearish + price near BB mid + RSI pullback + ADX strong
        elif htf_bearish and not price_vs_bb_upper and rsi_pullback_short and trend_strong:
            new_signal = -SIZE_ENTRY
        
        # Path 3: HMA crossover + HTF bearish + volume confirmation
        elif htf_bearish and fast_below_slow and hma_falling and vol_spike:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Range regime mean reversion - BB upper + RSI overbought + ADX weak
        elif regime_range and price_vs_bb_upper and rsi_overbought and trend_weak:
            new_signal = -SIZE_ENTRY
        
        # Path 5: BB rejection - price crosses below BB upper from above + RSI falling
        elif price_vs_bb_upper and i > 0 and close[i-1] > bb_upper[i-1] and rsi[i] < rsi[i-1]:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
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
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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