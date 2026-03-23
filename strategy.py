#!/usr/bin/env python3
"""
Experiment #017: 1d Primary + 4h HTF — SuperTrend + RSI-Streak + ADX Regime

Hypothesis: SuperTrend provides cleaner trend signals than HMA/KAMA crossovers (which failed).
RSI-Streak (Connors component) captures momentum exhaustion better than plain RSI.
ADX filter ensures we only trade when trend has strength (avoids chop whipsaw).
4h HTF gives better alignment than 1w for daily entries (1w too slow for 1d trading).

Key innovations vs failed strategies:
- SuperTrend (never tried) vs HMA/KAMA crossovers (failed in #003, #006, #015)
- RSI-Streak component vs plain RSI (more sensitive to momentum shifts)
- 4h HTF vs 1w HTF (better signal frequency for daily trading)
- ADX strength filter (avoids entering weak trends that reverse)

Why this should work:
- SuperTrend = ATR-based trend following, adapts to volatility automatically
- RSI-Streak = counts consecutive up/down days, catches exhaustion at 3-5 days
- ADX > 20 = ensures trend has momentum (avoids ranging whipsaw)
- 4h HTF bias = trade with higher timeframe trend direction
- Position size 0.25-0.30 = conservative for 77% crash scenarios

Target: 25-40 trades/year on 1d, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_rsi_streak_adx_4h_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate SuperTrend indicator.
    Returns: supertrend_values, supertrend_direction (1=uptrend, -1=downtrend)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Basic Upper/Lower Bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # SuperTrend values
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = uptrend (price above ST), -1 = downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = -1
    
    for i in range(1, n):
        if direction[i-1] == 1:
            # Previous trend was up
            if close[i] > lower_band[i]:
                supertrend[i] = max(supertrend[i-1], lower_band[i])
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previous trend was down
            if close[i] < upper_band[i]:
                supertrend[i] = min(supertrend[i-1], upper_band[i])
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_rsi_streak(close, period=14):
    """
    Calculate RSI-Streak component (from Connors RSI).
    Counts consecutive up/down days and converts to 0-100 scale.
    """
    n = len(close)
    streak = np.zeros(n)
    
    # Calculate streak: consecutive up/down days
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to 0-100 scale (percentile-like)
    # Positive streak (up days) = high values, negative = low values
    rsi_streak = np.zeros(n)
    for i in range(period, n):
        # Look back over period to normalize streak
        lookback_streaks = streak[max(0, i-period+1):i+1]
        max_streak = np.max(np.abs(lookback_streaks)) + 1e-10
        rsi_streak[i] = 50.0 + (streak[i] / max_streak) * 50.0
    
    # Clamp to 0-100
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    return rsi_streak

def calculate_rsi(close, period=14):
    """Calculate standard RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100.0 * plus_dm_s / (atr + 1e-10)
    minus_di = 100.0 * minus_dm_s / (atr + 1e-10)
    
    # DX
    di_sum = plus_di + minus_di + 1e-10
    dx = 100.0 * np.abs(plus_di - minus_di) / di_sum
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) for HTF trend."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi_14 = calculate_rsi(close, period=14)
    rsi_streak = calculate_rsi_streak(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(supertrend[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H MACRO BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20.0  # Trend has momentum
        adx_weak = adx_14[i] < 18.0    # Range/weak trend
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === RSI-STREAK EXTREMES (Connors component) ===
        streak_oversold = rsi_streak[i] < 25.0  # Many consecutive down days
        streak_overbought = rsi_streak[i] > 75.0  # Many consecutive up days
        
        # === VOLATILITY FILTER ===
        vol_normal = atr_7[i] < atr_14[i] * 1.3  # Not in extreme vol spike
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Supertrend bullish + ADX strong + 4h bias confirms ---
        if st_bullish and adx_strong:
            # Primary: Price above 4h HMA + RSI not overbought
            if price_above_hma_4h and rsi_14[i] < 70:
                new_signal = POSITION_SIZE
            # Secondary: RSI-streak oversold (momentum exhaustion bounce)
            elif streak_oversold and vol_normal:
                new_signal = POSITION_SIZE
            # Tertiary: RSI oversold + ADX rising
            elif rsi_oversold and adx_14[i] > adx_14[i-1]:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: Supertrend bearish + ADX strong + 4h bias confirms ---
        elif st_bearish and adx_strong:
            # Primary: Price below 4h HMA + RSI not oversold
            if price_below_hma_4h and rsi_14[i] > 30:
                new_signal = -POSITION_SIZE
            # Secondary: RSI-streak overbought (momentum exhaustion drop)
            elif streak_overbought and vol_normal:
                new_signal = -POSITION_SIZE
            # Tertiary: RSI overbought + ADX rising
            elif rsi_overbought and adx_14[i] > adx_14[i-1]:
                new_signal = -POSITION_SIZE
        
        # --- WEAK ADX (RANGE): Mean reversion plays ---
        elif adx_weak:
            # Long: RSI-streak very oversold in range
            if streak_oversold and rsi_14[i] < 40:
                new_signal = POSITION_SIZE * 0.5  # Smaller size in range
            # Short: RSI-streak very overbought in range
            elif streak_overbought and rsi_14[i] > 60:
                new_signal = -POSITION_SIZE * 0.5  # Smaller size in range
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if SuperTrend flips bearish
        if in_position and position_side > 0 and st_bearish:
            new_signal = 0.0
        
        # Exit short if SuperTrend flips bullish
        if in_position and position_side < 0 and st_bullish:
            new_signal = 0.0
        
        # Exit if 4h trend strongly opposes position
        if in_position and position_side > 0 and price_below_hma_4h and adx_strong:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_4h and adx_strong:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals