#!/usr/bin/env python3
"""
Experiment #009: 4h Fisher-Choppiness Regime Adaptive with 1d HMA Bias

Hypothesis: After 8 consecutive failures, the pattern is clear - strategies need
stronger HTF bias and better regime detection. This strategy combines:

1. 1D HMA trend bias: More stable than 4h HMA, filters out counter-trend trades
   that destroy performance in 2022 crash and 2025 bear market.

2. FISHER TRANSFORM (Ehlers): Superior reversal detection vs RSI at 4h timeframe.
   Entry when Fisher crosses -1.5 (long) or +1.5 (short) from extreme values.
   Less whipsaw than RSI in range markets.

3. CHOPPINESS INDEX regime filter: CHOP<38.2=trend (use breakouts), CHOP>61.8=range
   (use mean reversion). Prevents using wrong strategy in wrong regime.

4. DONCHIAN breakout confirmation: For trend regime, requires price to break
   20-bar Donchian channel with volume confirmation.

5. ATR trailing stoploss: 2.5*ATR to give trades room while protecting capital.

Why this should beat failed strategies (#001-#008 all Sharpe≤0):
- 1d HMA = more stable trend filter than 4h (proven in research)
- Fisher Transform > RSI for 4h reversals (Ehlers research)
- Regime-adaptive = works in bull (2021), crash (2022), bear (2025)
- Looser Fisher thresholds (-1.5/+1.5 vs -2.0/+2.0) = ensures ≥10 trades
- Position sizing 0.25-0.30 discrete = minimizes fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete, ATR-scaled
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_1d_hma_regime_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = normalized price
    Excellent for catching reversals at extremes (-2.0 to +2.0)
    """
    close_s = pd.Series(close)
    
    # Normalize price to -1 to +1 range using highest high / lowest low
    highest = close_s.rolling(window=period, min_periods=period).max()
    lowest = close_s.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    price_range = highest - lowest
    price_range = price_range.replace(0, 0.001)
    
    # Normalize: (close - lowest) / (highest - lowest) * 2 - 1
    x = ((close_s - lowest) / price_range * 2 - 1).clip(-0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True range for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Price range
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.inf)
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

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
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher(close, 9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio for volatility scaling
    atr_30 = calculate_atr(high, low, close, 30)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = np.inf
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        if np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 38.2
        is_range_regime = chop_14[i] > 61.8
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.7 * vol_sma[i]
        
        # === VOLATILITY SCALING ===
        atr_ratio = atr_14[i] / atr_30[i] if atr_30[i] > 0 else 1.0
        atr_ratio = np.clip(atr_ratio, 0.5, 2.0)
        size_multiplier = 1.0 / atr_ratio
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.20, 0.35)
        current_size = np.round(current_size * 4) / 4  # Discrete: 0.20, 0.25, 0.30, 0.35
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crossing above -1.5 from below (bullish reversal)
        fisher_long_cross = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        # Fisher crossing below +1.5 from above (bearish reversal)
        fisher_short_cross = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Fisher at extreme (for range regime mean reversion)
        fisher_oversold = fisher[i] < -1.6
        fisher_overbought = fisher[i] > 1.6
        
        # === DONCHIAN BREAKOUT (for trend regime) ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === BOLLINGER BAND MEAN REVERSION (for range regime) ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005
        near_bb_upper = close[i] > bb_upper[i] * 0.995
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: TREND REGIME - Breakout with HTF bias + volume
        if is_trend_regime:
            # Long: Donchian breakout + 1d bullish bias + volume confirmed
            if breakout_long and bull_bias and volume_confirmed:
                new_signal = current_size
            
            # Short: Donchian breakout + 1d bearish bias + volume confirmed
            elif breakout_short and bear_bias and volume_confirmed:
                new_signal = -current_size
        
        # MODE 2: RANGE REGIME - Fisher mean reversion + BB extremes
        elif is_range_regime:
            # Long: Fisher oversold + near BB lower + 1d bias not strongly bearish
            if fisher_oversold and near_bb_lower:
                if not bear_bias or chop_14[i] > 55:
                    new_signal = current_size
            
            # Short: Fisher overbought + near BB upper + 1d bias not strongly bullish
            elif fisher_overbought and near_bb_upper:
                if not bull_bias or chop_14[i] > 55:
                    new_signal = -current_size
        
        # MODE 3: TRANSITION REGIME - Fisher crossover signals with HTF bias
        else:
            # Long: Fisher crosses above -1.5 + 1d bias not bearish
            if fisher_long_cross and not bear_bias:
                new_signal = current_size
            
            # Short: Fisher crosses below +1.5 + 1d bias not bullish
            elif fisher_short_cross and not bull_bias:
                new_signal = -current_size
        
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
                if close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d bias turns bearish with trending regime
            if position_side > 0 and bear_bias and is_trend_regime:
                trend_reversal = True
            # Exit short if 1d bias turns bullish with trending regime
            if position_side < 0 and bull_bias and is_trend_regime:
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
                lowest_price = close[i] if position_side < 0 else np.inf
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else np.inf
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = np.inf
        
        signals[i] = new_signal
    
    return signals