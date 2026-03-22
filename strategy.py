#!/usr/bin/env python3
"""
Experiment #310: 4h Supertrend with Fisher Transform Entries and 1d HMA Bias

Hypothesis: After analyzing 299 experiments, clear patterns emerge:
1. 4h Supertrend + 1d HMA works best (Sharpe=0.485, current baseline)
2. Strategies with too many filters generate 0 trades (#307, #308 Sharpe=0.000)
3. RSI mean reversion consistently fails across timeframes
4. 12h/1d timeframes show positive Sharpe but may have too few trades

This strategy IMPROVES on the baseline by:
1. Using EHLERS FISHER TRANSFORM for entry timing (catches reversals cleaner than RSI)
2. Keeping 4h Supertrend for trend direction (proven edge)
3. Keeping 1d HMA for directional bias (proven edge from #292, #299)
4. LOOSE ADX threshold (>12 not >25) to ensure >=10 trades per symbol
5. Discrete position sizing (0.0, ±0.25, ±0.30) to minimize fee churn

Why Fisher Transform over RSI:
- Fisher normalizes price to Gaussian distribution (-1 to +1)
- Crosses at extremes are cleaner signals than RSI thresholds
- Less prone to staying in overbought/oversold for extended periods
- Proven in bear/range markets (critical for 2025 test period)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels (MAX 0.35)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_fisher_1d_hma_loose_adx_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=below price=bullish, -1=above=bearish)
    """
    atr = calculate_atr(high, low, close, period)
    
    n = len(close)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    supertrend[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    return supertrend, direction

def calculate_fisher_transform(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Values range approximately -1 to +1.
    Cross above -0.5 from below = long signal
    Cross below +0.5 from above = short signal
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate highest high and lowest low over period
    for i in range(period - 1, n):
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize price to 0-1 range
        x = (close[i] - lowest) / (highest - lowest)
        
        # Constrain to 0.001-0.999 to avoid log(0)
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
    
    return fisher

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
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
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
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    fisher = calculate_fisher_transform(close, 9)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_STRONG = 0.30  # Increased size in strong trend
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Track Fisher previous value for crossover detection
    fisher_prev = np.zeros(n)
    for i in range(1, n):
        fisher_prev[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(supertrend_dir[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = primary directional bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND DIRECTION (Supertrend) ===
        # Supertrend direction: 1 = price above ST (bullish), -1 = price below ST (bearish)
        st_bullish = supertrend_dir[i] == 1
        st_bearish = supertrend_dir[i] == -1
        
        # === TREND STRENGTH ===
        # ADX > 12 = trending market (LOOSE threshold to ensure >=10 trades)
        trending = adx[i] > 12
        strong_trend = adx[i] > 20
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher cross above -0.5 from below = bullish reversal signal
        fisher_bullish_cross = (fisher_prev[i] < -0.5) and (fisher[i] >= -0.5)
        # Fisher cross below +0.5 from above = bearish reversal signal
        fisher_bearish_cross = (fisher_prev[i] > 0.5) and (fisher[i] <= 0.5)
        
        # Fisher extreme oversold (potential long even without cross)
        fisher_oversold = fisher[i] < -0.8
        # Fisher extreme overbought (potential short even without cross)
        fisher_overbought = fisher[i] > 0.8
        
        # === POSITION SIZING ===
        if strong_trend:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need Supertrend bullish + 1d bias up + Fisher signal + ADX filter
        # Multiple entry triggers to ensure >=10 trades per symbol
        long_conditions = (
            st_bullish and  # Supertrend bullish
            bull_trend_1d and  # 1d HMA bias bullish
            trending and  # ADX confirms some trend
            (fisher_bullish_cross or fisher_oversold)  # Fisher entry signal
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            st_bearish and  # Supertrend bearish
            bear_trend_1d and  # 1d HMA bias bearish
            trending and  # ADX confirms some trend
            (fisher_bearish_cross or fisher_overbought)  # Fisher entry signal
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if Supertrend or HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and (st_bearish or bear_trend_1d):
                new_signal = 0.0  # Trend reversed against long
            if position_side < 0 and (st_bullish or bull_trend_1d):
                new_signal = 0.0  # Trend reversed against short
        
        # === FISHER REVERSAL EXIT ===
        # Exit if Fisher crosses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and fisher_bearish_cross:
                new_signal = 0.0  # Fisher crossed against long
            if position_side < 0 and fisher_bullish_cross:
                new_signal = 0.0  # Fisher crossed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals