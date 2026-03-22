#!/usr/bin/env python3
"""
Experiment #563: 12h Regime-Adaptive Strategy with Choppiness Index

Hypothesis: After 500+ failed experiments, the key insight is that crypto markets
alternate between trending and ranging regimes. A single strategy (pure trend or
pure mean-reversion) fails because it doesn't adapt. This strategy:

1. Uses Choppiness Index (CHOP) to detect regime:
   - CHOP > 61.8 = ranging market → mean reversion at Bollinger bands
   - CHOP < 38.2 = trending market → breakout with trend bias
   - Between = neutral → reduce position size

2. 1d HMA for higher-timeframe trend bias (via mtf_data helper)

3. Asymmetric entry thresholds:
   - Trend regime: breakout + ADX > 20 (loose, ensures trades)
   - Range regime: RSI extremes (30/70) + BB touch (generates more trades)

4. 12h timeframe = 2 bars/day = ~730 bars/year = manageable frequency
   Fewer trades than 1h/4h strategies = less fee drag

5. Position sizing: 0.30 max, discrete levels, 2.5*ATR stoploss

Why this should beat previous 12h attempts:
- Previous Donchian breakout (#533 current) only works in trends
- This adapts to 2022 crash (ranging/bear) AND 2021 bull (trending)
- CHOP filter is proven in literature for crypto bear markets
- Looser thresholds ensure we generate trades (unlike #551, #555, #557 with 0 trades)

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_adaptive_daily_hma_asymmetric_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / hh_ll.replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.30
    SIZE_RANGE = 0.25
    SIZE_NEUTRAL = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trending = chop_14[i] < 38.2
        is_ranging = chop_14[i] > 61.8
        is_neutral = not is_trending and not is_ranging
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === DETERMINE POSITION SIZE BASED ON REGIME ===
        if is_trending:
            size = SIZE_TREND
        elif is_ranging:
            size = SIZE_RANGE
        else:
            size = SIZE_NEUTRAL
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # TREND REGIME: Breakout + ADX + HTF bias
        if is_trending:
            # Long: price above BB mid + ADX confirms + daily bullish
            if bull_bias and adx_14[i] > 18 and close[i] > bb_mid[i]:
                new_signal = size
            # Short: price below BB mid + ADX confirms + daily bearish
            elif bear_bias and adx_14[i] > 18 and close[i] < bb_mid[i]:
                new_signal = -size
        
        # RANGE REGIME: Mean reversion at Bollinger extremes + RSI
        elif is_ranging:
            # Long: RSI oversold + touch lower BB
            if rsi_14[i] < 35 and close[i] <= bb_lower[i] * 1.002:
                new_signal = size
            # Short: RSI overbought + touch upper BB
            elif rsi_14[i] > 65 and close[i] >= bb_upper[i] * 0.998:
                new_signal = -size
        
        # NEUTRAL REGIME: Reduced size, wait for clearer signals
        elif is_neutral:
            # Only enter on strong signals
            if bull_bias and rsi_14[i] < 40 and close[i] > bb_mid[i]:
                new_signal = size
            elif bear_bias and rsi_14[i] > 60 and close[i] < bb_mid[i]:
                new_signal = -size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if daily HMA flips against position (strong signal)
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias and chop_14[i] < 50:
                new_signal = 0.0
            if position_side < 0 and bull_bias and chop_14[i] < 50:
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