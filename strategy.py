#!/usr/bin/env python3
"""
Experiment #375: 1h Keltner Squeeze Breakout with 4h HMA Trend + Volume Filter

Hypothesis: After 374 failed experiments, the key insight is that successful strategies need:
1. Higher timeframe trend filter (4h HMA proven in best strategies)
2. Different channel type (Keltner vs Bollinger/Donchian - less tested)
3. Volume confirmation (filters false breakouts)
4. ADX regime filter (avoid chop)
5. Proper stoploss (ATR trailing)

Why 1h Keltner might work:
- Keltner Channels (EMA + ATR bands) are less tested than Bollinger/Donchian
- Volume confirmation adds edge (most failed strategies ignore volume)
- 4h HMA provides stable trend bias (proven in mtf_4h_regime_chop best strategy)
- 1h timeframe captures intraday moves without excessive noise
- Should generate 30-60 trades/year (enough for stats, not too many for fees)

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data helper
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_keltner_4h_hma_volume_adx_atr_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 10:
        return adx
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_keltner_channels(high, low, close, ema_period=20, atr_period=14, multiplier=2.0):
    """
    Calculate Keltner Channels.
    Middle = EMA(close, 20)
    Upper = Middle + multiplier * ATR(14)
    Lower = Middle - multiplier * ATR(14)
    """
    close_s = pd.Series(close)
    middle = close_s.ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = middle + multiplier * atr
    lower = middle - multiplier * atr
    
    return upper, lower, middle

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    keltner_upper, keltner_lower, keltner_middle = calculate_keltner_channels(high, low, close, 20, 14, 2.0)
    volume_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(volume_ma[i]) or volume_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === ADX TREND STRENGTH ===
        trending_market = adx[i] > 20
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above average for breakout confirmation
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # === KELTNER BREAKOUT SIGNALS ===
        # Long breakout: price breaks above Keltner upper
        long_breakout = close[i] > keltner_upper[i-1] if i > 0 else False
        
        # Short breakout: price breaks below Keltner lower
        short_breakout = close[i] < keltner_lower[i-1] if i > 0 else False
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Breakout + 4h bullish bias + ADX confirms + Volume confirms
        if long_breakout and bull_trend_4h and trending_market and volume_confirmed:
            new_signal = SIZE
        
        # SHORT ENTRY: Breakout + 4h bearish bias + ADX confirms + Volume confirms
        elif short_breakout and bear_trend_4h and trending_market and volume_confirmed:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === ADX DROPS BELOW THRESHOLD ===
        # Exit if market becomes ranging (ADX < 18 with hysteresis)
        if in_position and adx[i] < 18:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals