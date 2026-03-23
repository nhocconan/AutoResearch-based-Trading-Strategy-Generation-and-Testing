#!/usr/bin/env python3
"""
Experiment #056: 12h Primary + 1d HTF — Simplified Trend Following

Hypothesis: Previous 12h/1d strategies failed due to excessive filters causing 0 trades.
This strategy uses SIMPLIFIED logic to ensure trade generation:

1. 1d HMA(21) SLOPE for major trend direction (not just price position)
2. 12h HMA(8/21) crossover for entry timing (faster than price vs HMA)
3. RSI(14) with 40/60 thresholds (not extreme 20/80 - allows more entries)
4. ADX(14) > 18 for trend confirmation (lower threshold than typical 25)
5. ATR(14) stoploss at 2.0x (tighter than 2.5x for better risk control)
6. NO volume filter, NO session filter (12h doesn't need these)
7. Position size: 0.28 discrete (balanced between risk and opportunity)

Why this should work:
- 12h timeframe naturally limits trades to 20-50/year
- HMA crossover is more responsive than price vs HMA
- RSI 40/60 thresholds catch more opportunities than 20/80 extremes
- ADX > 18 is achievable (ADX > 25 too rare in crypto ranges)
- Simpler logic = more trades = better statistics
- 1d HMA slope prevents counter-trend trades in strong trends

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete
Stoploss: 2.0 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crossover_rsi_adx_1d_v1"
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # HMA crossover signals (fast vs slow)
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        # HMA slope > 0 = bullish bias (prefer longs)
        # HMA slope < 0 = bearish bias (prefer shorts)
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0
        trend_1d_bearish = hma_1d_slope_aligned[i] < 0
        
        # Also check price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA CROSSOVER ===
        # Fast HMA(8) crosses above Slow HMA(21) = bullish crossover
        # Fast HMA(8) crosses below Slow HMA(21) = bearish crossover
        hma_bullish_cross = hma_8[i] > hma_21[i] and hma_8[i-1] <= hma_21[i-1]
        hma_bearish_cross = hma_8[i] < hma_21[i] and hma_8[i-1] >= hma_21[i-1]
        
        # Current HMA alignment
        hma_aligned_bullish = hma_8[i] > hma_21[i]
        hma_aligned_bearish = hma_8[i] < hma_21[i]
        
        # === ADX TREND STRENGTH ===
        # ADX > 18 = trending market (allow trend following)
        # ADX <= 18 = ranging market (reduce position or skip)
        trend_strong = adx_14[i] > 18
        
        # === RSI MOMENTUM ===
        # RSI > 50 = bullish momentum
        # RSI < 50 = bearish momentum
        # Use 40/60 thresholds for entries (not extreme 20/80)
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_entry_long = rsi_14[i] > 40  # Allow entries above 40
        rsi_entry_short = rsi_14[i] < 60  # Allow entries below 60
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in weak trends (ADX < 25)
        if adx_14[i] < 25:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 1d bullish bias + 12h HMA aligned bullish + RSI confirmation
        if trend_1d_bullish or price_above_1d_hma:
            if hma_aligned_bullish and rsi_entry_long:
                # Strong entry on crossover
                if hma_bullish_cross and trend_strong:
                    new_signal = current_size
                # Entry on pullback in established trend
                elif hma_aligned_bullish and rsi_14[i] > 45 and rsi_14[i] < 65:
                    new_signal = current_size * 0.8
        
        # SHORT ENTRIES
        # Require: 1d bearish bias + 12h HMA aligned bearish + RSI confirmation
        if trend_1d_bearish or price_below_1d_hma:
            if hma_aligned_bearish and rsi_entry_short:
                # Strong entry on crossover
                if hma_bearish_cross and trend_strong:
                    new_signal = -current_size
                # Entry on pullback in established trend
                elif hma_aligned_bearish and rsi_14[i] > 35 and rsi_14[i] < 55:
                    new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~75 days on 12h), allow weaker entry
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and hma_aligned_bullish and rsi_14[i] > 45:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and hma_aligned_bearish and rsi_14[i] < 55:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend reverses bearish
            if position_side > 0 and trend_1d_bearish and hma_aligned_bearish:
                trend_reversal = True
            # Exit short if 1d trend reverses bullish
            if position_side < 0 and trend_1d_bullish and hma_aligned_bullish:
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