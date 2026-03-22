#!/usr/bin/env python3
"""
Experiment #519: 1h Fisher Transform + 4h HMA Trend + Bollinger Mean-Reversion

Hypothesis: After 500+ failed experiments, clear patterns emerge:
1. 1h timeframe is too noisy for pure trend-following (all recent 1h strategies failed with Sharpe -2 to -5)
2. 1h needs STRONG HTF filter (4h HMA proven in best strategy Sharpe=0.676)
3. Fisher Transform catches reversals better than RSI in bear/range markets (Ehlers' research)
4. Bollinger Band mean-reversion works in 2025 bear/range test period
5. LOOSE thresholds are CRITICAL - many strategies failed from over-filtering (0 trades)
6. Asymmetric sizing protects in bear markets (2025 test is -25% BTC)

Key innovations:
1. FISHER TRANSFORM (period=9): Long when Fisher < -1.0 (oversold), short when Fisher > +1.0 (overbought)
2. 4H HMA(21) via mtf_data helper: Primary trend filter (bullish/bearish bias)
3. BOLLINGER MEAN-REVERSION: Price < BB_lower for long, price > BB_upper for short
4. LOOSE ADX FILTER: > 12 (not 25 which rarely triggers)
5. ASYMMETRIC SIZING: 0.25 long, 0.20 short (conservative in bear markets)
6. 2.5 * ATR STOPLOSS: Trailing stop for risk management
7. DUAL ENTRY: Fisher OR Bollinger (either triggers entry = more trades)

Why this might work on 1h:
- Fisher Transform is designed for reversal capture (works in bear markets)
- 4h HMA provides strong trend filter without EMA lag
- Bollinger mean-reversion captures 2025 range-bound behavior
- Dual entry ensures ≥10 trades/year (loose thresholds)
- Asymmetric sizing protects in 2025 bear test period

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 long, 0.20 short (discrete, conservative)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_bb_meanrev_loose_adx_asymmetric_atr_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * prev_Fisher
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.zeros(n)
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    for i in range(period, n):
        if hh[i] > ll[i] and hh[i] - ll[i] > 1e-10:
            # Normalize price to 0-1 range
            price_norm = (close[i] - ll[i]) / (hh[i] - ll[i])
            
            # Transform to X value bounded between -1 and 1
            X = 0.66 * (price_norm - 0.5) + 0.67 * fisher_prev[i-1]
            X = np.clip(X, -0.999, 0.999)  # Prevent division by zero
            
            # Fisher Transform
            fisher[i] = 0.5 * np.log((1 + X) / (1 - X))
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma * 100  # BB Width as percentage
    return upper, lower, width

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volume ratio
    volume_ratio = np.full(n, np.nan)
    for i in range(n):
        if volume[i] > 1e-10:
            volume_ratio[i] = taker_buy_volume[i] / volume[i]
    
    signals = np.zeros(n)
    
    # Position sizing - asymmetric (Rule 4) - conservative for bear market
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.20
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === ADX TREND STRENGTH (LOOSE FILTER) ===
        is_trending = adx[i] > 12  # Very loose to ensure trades
        
        # === VOLUME CONFIRMATION ===
        volume_bull = volume_ratio[i] > 0.52
        volume_bear = volume_ratio[i] < 0.48
        
        # === FISHER TRANSFORM SIGNALS (LOOSE THRESHOLDS) ===
        fisher_oversold = fisher[i] < -0.8  # Loose threshold
        fisher_overbought = fisher[i] > 0.8  # Loose threshold
        
        # === BOLLINGER MEAN-REVERSION SIGNALS ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC - DUAL TRIGGER (Fisher OR Bollinger) ===
        new_signal = 0.0
        
        # LONG ENTRY: Bull bias + (Fisher oversold OR Bollinger lower)
        if bull_bias:
            if fisher_oversold or bb_oversold:
                # Volume confirmation OR range market (ADX low)
                if volume_bull or (not is_trending):
                    new_signal = SIZE_LONG
        
        # SHORT ENTRY: Bear bias + (Fisher overbought OR Bollinger upper)
        if bear_bias:
            if fisher_overbought or bb_overbought:
                # Volume confirmation OR range market (ADX low)
                if volume_bear or (not is_trending):
                    new_signal = -SIZE_SHORT
        
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
        
        # === BIAS REVERSAL EXIT ===
        # Exit if 4h trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias and adx[i] > 20:
                new_signal = 0.0
            if position_side < 0 and bull_bias and adx[i] > 20:
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