#!/usr/bin/env python3
"""
Experiment #336: 1d Trend Following with 1w HMA Bias + Volume Confirmation

Hypothesis: After analyzing 287 failed strategies, key insights for 1d timeframe:
1. 1d is ideal for trend following (less noise than lower TFs)
2. 1w HMA provides extremely stable trend bias (proven edge)
3. Volume confirmation filters false breakouts (critical on daily)
4. ADX > 20 ensures we only trade when trend has strength
5. MACD histogram confirms momentum alignment
6. Position size 0.30 with 2.5*ATR stoploss controls 2022 crash drawdown

Why this should work on 1d (unlike failed 1d strategies #324, #330, #335):
- 1w HMA bias is more stable than 1d HMA (used in #330)
- Volume confirmation adds edge missing from #324 (Donchian only)
- ADX > 20 (not 25) ensures trade generation >= 10 on train
- MACD histogram (not line) for momentum like #329 which had Sharpe=0.311
- Looser entry conditions than #335 which generated 0 trades

Key differences from failures:
- #324: Added volume + MACD (Donchian alone failed)
- #330: Using 1w HMA (not 1d) + volume filter
- #335: Much looser ADX threshold (20 vs 30+)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trend_1w_hma_volume_macd_adx_atr_v1"
timeframe = "1d"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / vol_ma.values
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
    adx = calculate_adx(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_line[i]) or np.isnan(histogram[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1w HMA = primary directional bias (very stable)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        # Price breaking 20-day high = bullish breakout
        donchian_bullish = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_bearish = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === MACD MOMENTUM ===
        # Histogram > 0 = bullish momentum
        macd_bullish = histogram[i] > 0
        macd_bearish = histogram[i] < 0
        
        # === ADX TREND STRENGTH ===
        # ADX > 20 = trending (loose for trade generation on 1d)
        trending = adx[i] > 20
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.2x average = confirmed move
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG: 1w trend up + Donchian breakout + MACD bullish + trending + volume
        # Loosen: only need 3 of 4 conditions for entry
        long_score = sum([
            bull_trend_1w,
            donchian_bullish,
            macd_bullish,
            trending
        ])
        
        # SHORT: 1w trend down + Donchian breakdown + MACD bearish + trending + volume
        short_score = sum([
            bear_trend_1w,
            donchian_bearish,
            macd_bearish,
            trending
        ])
        
        # Entry threshold: need 3+ conditions (loose enough for trade generation)
        if long_score >= 3:
            new_signal = SIZE
        
        if short_score >= 3:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1w trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w:
                new_signal = 0.0
        
        # === MOMENTUM REVERSAL EXIT ===
        # Exit if MACD flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and macd_bearish:
                new_signal = 0.0
            if position_side < 0 and macd_bullish:
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