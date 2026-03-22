#!/usr/bin/env python3
"""
Experiment #074: 4h Primary + 12h/1d HTF — Simplified Trend Following

Hypothesis: Previous strategies failed due to excessive filters (Choppiness, Connors, dual-regime)
causing too few trades or whipsaw losses. This strategy uses SIMPLIFIED trend logic:

1. 12h HMA(21) SLOPE for major trend bias (not just price position)
2. 4h HMA(8/21) crossover for entry timing (proven pattern from research)
3. RSI(14) with 45/55 thresholds (moderate, allows more entries than 20/80)
4. ADX(14) > 20 for trend confirmation (achievable threshold)
5. ATR(14) stoploss at 2.5x (standard trailing stop)
6. NO Choppiness Index, NO Connors RSI, NO complex regime detection
7. Position size: 0.30 discrete for strong signals, 0.20 for weaker
8. Minimum trade frequency safeguard (enter if no trade in 100 bars)

Why this should work:
- 4h timeframe naturally limits trades to 30-60/year
- HMA crossover is responsive but not whipsaw-prone like EMA
- RSI 45/55 catches momentum without waiting for extremes
- ADX > 20 is common enough in crypto trends
- Simpler logic = more trades = better statistics
- 12h HMA slope prevents counter-trend trades

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 max discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_crossover_rsi_adx_12h_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # HMA crossover signals (fast vs slow)
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            continue
        
        # === 12H TREND BIAS (MAJOR) ===
        # HMA slope > 0.5 = bullish bias (prefer longs)
        # HMA slope < -0.5 = bearish bias (prefer shorts)
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.5
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.5
        
        # Price vs 12h HMA for additional confirmation
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === 4H HMA CROSSOVER ===
        # Fast HMA(8) crosses above Slow HMA(21) = bullish crossover
        # Fast HMA(8) crosses below Slow HMA(21) = bearish crossover
        hma_bullish_cross = hma_8[i] > hma_21[i] and hma_8[i-1] <= hma_21[i-1]
        hma_bearish_cross = hma_8[i] < hma_21[i] and hma_8[i-1] >= hma_21[i-1]
        
        # Current HMA alignment
        hma_aligned_bullish = hma_8[i] > hma_21[i]
        hma_aligned_bearish = hma_8[i] < hma_21[i]
        
        # === ADX TREND STRENGTH ===
        # ADX > 20 = trending market (allow trend following)
        trend_strong = adx_14[i] > 20
        
        # === RSI MOMENTUM ===
        # RSI > 50 = bullish momentum
        # RSI < 50 = bearish momentum
        # Use 45/55 thresholds for entries (moderate, not extreme)
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_entry_long = rsi_14[i] > 45
        rsi_entry_short = rsi_14[i] < 55
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE if trend_strong else REDUCED_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Primary: 12h bullish + 4h HMA bullish cross + RSI confirmation + ADX
        if trend_12h_bullish or price_above_12h_hma:
            if hma_bullish_cross and rsi_entry_long:
                if trend_strong:
                    new_signal = BASE_SIZE
                else:
                    new_signal = REDUCED_SIZE
            # Pullback entry in established trend
            elif hma_aligned_bullish and rsi_14[i] > 45 and rsi_14[i] < 60:
                new_signal = REDUCED_SIZE
        
        # SHORT ENTRIES
        # Primary: 12h bearish + 4h HMA bearish cross + RSI confirmation + ADX
        if trend_12h_bearish or price_below_12h_hma:
            if hma_bearish_cross and rsi_entry_short:
                if trend_strong:
                    new_signal = -BASE_SIZE
                else:
                    new_signal = -REDUCED_SIZE
            # Pullback entry in established trend
            elif hma_aligned_bearish and rsi_14[i] > 40 and rsi_14[i] < 55:
                new_signal = -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~25 days on 4h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and hma_aligned_bullish and rsi_14[i] > 48:
                new_signal = REDUCED_SIZE * 0.8
            elif trend_12h_bearish and hma_aligned_bearish and rsi_14[i] < 52:
                new_signal = -REDUCED_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h trend reverses bearish
            if position_side > 0 and trend_12h_bearish and hma_aligned_bearish:
                trend_reversal = True
            # Exit short if 12h trend reverses bullish
            if position_side < 0 and trend_12h_bullish and hma_aligned_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals