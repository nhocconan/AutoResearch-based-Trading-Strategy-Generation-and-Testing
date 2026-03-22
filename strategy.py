#!/usr/bin/env python3
"""
Experiment #324: 1d Donchian Breakout with Weekly HMA Bias and Volume Confirmation

Hypothesis: After #312 and #318 failed on 1d with mean-reversion approaches, I'll try
pure trend-following with LOOSE entry conditions. Analysis shows:
1. Mean reversion fails on 1d (RSI/CRSI strategies all negative Sharpe)
2. Trend following works better on daily timeframes
3. The 4h regime chop strategy (Sharpe=0.676) proves regime detection + trend works
4. Need VERY LOOSE conditions to ensure >=10 trades on daily (fewer bars)

This strategy uses:
1. Donchian(20) breakout on 1d for entry timing (classic trend following)
2. 1w HMA(21) for meta-trend bias (SOFT - boosts confidence, not hard requirement)
3. ADX(14)>15 for minimal trend confirmation (loose threshold)
4. Volume > 1.2x 20-day avg for breakout confirmation (avoids fake breakouts)
5. ATR(14) trailing stoploss at 2.5x (proven from successful strategies)
6. Position exit on Donchian midpoint breach (trend weakening)

Key differences from failed #312/#318:
- Trend following instead of mean reversion (better for 1d)
- Donchian breakout instead of EMA crossover (cleaner signals)
- Volume confirmation (reduces fake breakouts)
- ADX threshold very loose (15 not 25) for trade generation
- Weekly HMA is SOFT filter only (doesn't block entries)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_hma_volume_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower/mid)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

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
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (SOFT) ===
        # 1w HMA = meta-trend confirmation (boosts size but NOT required for entry)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 15 = minimal trending (loose for trade generation)
        trending = adx[i] > 15
        strong_trend = adx[i] > 25
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.2x 20-day average confirms breakout validity
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # === DONCHIAN BREAKOUT ===
        # Price breaks above Donchian upper = bullish breakout
        breakout_bull = close[i] > donchian_upper[i-1]  # Use previous bar's upper
        # Price breaks below Donchian lower = bearish breakout
        breakout_bear = close[i] < donchian_lower[i-1]  # Use previous bar's lower
        
        # Check for actual breakout (not just above/below)
        if i > 0:
            breakout_bull = breakout_bull and (close[i-1] <= donchian_upper[i-1])
            breakout_bear = breakout_bear and (close[i-1] >= donchian_lower[i-1])
        
        # === POSITION SIZING ===
        if strong_trend and bull_trend_1w:
            position_size = SIZE_STRONG
        elif strong_trend and bear_trend_1w:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS (LOOSE for >=10 trades) ===
        new_signal = 0.0
        
        # LONG: Donchian breakout + volume + ADX trending
        # 1w bias is SOFT - only affects size, not entry
        long_conditions = (
            breakout_bull and
            volume_confirmed and
            trending
        )
        
        # SHORT: Donchian breakout + volume + ADX trending
        short_conditions = (
            breakout_bear and
            volume_confirmed and
            trending
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
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
        
        # === DONCHIAN MIDPOINT EXIT (trend weakening) ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and close[i] < donchian_mid[i]:
                new_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
                new_signal = 0.0
        
        # === WEEKLY TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            # Only exit if 1w trend strongly reverses AND we're in loss
            if position_side > 0 and bear_trend_1w:
                if close[i] < entry_price:
                    new_signal = 0.0
            if position_side < 0 and bull_trend_1w:
                if close[i] > entry_price:
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