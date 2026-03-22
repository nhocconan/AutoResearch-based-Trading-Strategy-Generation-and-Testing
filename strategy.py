#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian-ADX Breakout with 1D HMA Trend Bias

Hypothesis: After 4 failed experiments, the pattern shows lower TFs (15m-1h) suffer
from noise and fee drag, while pure trend strategies fail in bear/range markets.
This 12h strategy combines:

1. 1D HMA trend bias: Ultra-stable HTF direction filter. Only long if price>1d_HMA,
   only short if price<1d_HMA. Much more stable than 4h for 12h primary TF.

2. Donchian(20) breakout: Captures sustained moves with clear entry/exit. Works
   well on 12h timeframe (less fakeouts than lower TFs).

3. ADX(14) filter: ADX>25 confirms trend strength, prevents entries in chop.
   Critical for avoiding 2022-style whipsaw losses.

4. Bollinger Band Width regime: Narrow BB = squeeze = breakout potential.
   Enter when BBW < 30th percentile (compression before expansion).

5. Volume confirmation: Breakout volume > 0.8*20bar_avg filters fakeouts.

6. ATR-based sizing: Reduce position when vol is high (protects in 2022 crash).

Why 12h should beat failed strategies:
- 12h has 2x fewer trades than 4h, 8x fewer than 1h = much less fee drag
- 1D HMA bias is extremely stable (changes rarely = less churn)
- Donchian+ADX combo proven on daily/weekly charts in literature
- BBW squeeze filter = enter only when volatility is primed to expand
- Target 25-40 trades/year = optimal for 12h (Rule 10)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete, ATR-scaled
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_adx_1d_hma_bbw_vol_atr_v1"
timeframe = "12h"
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
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.inf))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.inf))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma * 100  # Bandwidth as percentage
    
    return upper.values, lower.values, sma.values, bb_width.values

def calculate_bbw_percentile(bb_width, lookback=100):
    """Calculate rolling percentile of BB Width for squeeze detection."""
    bbw_s = pd.Series(bb_width)
    bbw_percentile = bbw_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if (x.max() - x.min()) > 0 else 50
    )
    return bbw_percentile.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger(close, 20, 2.0)
    bbw_pct = calculate_bbw_percentile(bb_width, 100)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        if np.isnan(bbw_pct[i]):
            continue
        
        # === 1D HMA TREND BIAS (Ultra-stable HTF filter) ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === ADX TREND STRENGTH ===
        is_trending = adx_14[i] > 25
        is_strong_trend = adx_14[i] > 30
        
        # === BOLLINGER BAND WIDTH SQUEEZE ===
        # BBW percentile < 30 = compression (breakout potential)
        is_squeeze = bbw_pct[i] < 30
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout long: price crosses above previous Donchian upper
        breakout_long = False
        if i > 0 and not np.isnan(donchian_upper[i-1]):
            breakout_long = (close[i] > donchian_upper[i-1]) and (close[i-1] <= donchian_upper[i-1])
        
        # Breakout short: price crosses below previous Donchian lower
        breakout_short = False
        if i > 0 and not np.isnan(donchian_lower[i-1]):
            breakout_short = (close[i] < donchian_lower[i-1]) and (close[i-1] >= donchian_lower[i-1])
        
        # === ATR-BASED POSITION SIZING ===
        # Reduce size when volatility is high (protect in crashes)
        if i > 100:
            atr_median = np.nanmedian(atr_14[100:i])
            if atr_median > 0:
                atr_ratio = atr_14[i] / atr_median
                atr_ratio = np.clip(atr_ratio, 0.5, 2.5)
                size_multiplier = 1.0 / atr_ratio
            else:
                size_multiplier = 1.0
        else:
            size_multiplier = 1.0
        
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: SQUEEZE BREAKOUT (highest conviction)
        # BB squeeze + Donchian breakout + ADX confirming + HTF bias + volume
        if is_squeeze and volume_confirmed:
            # Long: squeeze breakout + bullish 1D bias + ADX rising
            if breakout_long and bull_bias and is_trending:
                new_signal = current_size
            
            # Short: squeeze breakout + bearish 1D bias + ADX rising
            elif breakout_short and bear_bias and is_trending:
                new_signal = -current_size
        
        # MODE 2: TREND CONTINUATION (strong ADX + HTF bias)
        elif is_strong_trend and volume_confirmed:
            # Long: strong trend + bullish bias + Donchian breakout
            if breakout_long and bull_bias:
                new_signal = current_size
            
            # Short: strong trend + bearish bias + Donchian breakout
            elif breakout_short and bear_bias:
                new_signal = -current_size
        
        # MODE 3: STANDARD BREAKOUT (ADX moderate + HTF bias)
        elif is_trending and volume_confirmed:
            # Long: trending + bullish bias + breakout
            if breakout_long and bull_bias:
                new_signal = current_size * 0.8  # Smaller size for standard breakout
            
            # Short: trending + bearish bias + breakout
            elif breakout_short and bear_bias:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1D bias turns bearish with strong ADX
            if position_side > 0 and bear_bias and adx_14[i] > 25:
                trend_reversal = True
            # Exit short if 1D bias turns bullish with strong ADX
            if position_side < 0 and bull_bias and adx_14[i] > 25:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals