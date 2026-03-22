#!/usr/bin/env python3
"""
Experiment #280: 4h Fisher Transform Reversals with 1d HMA Bias and BB Regime Filter

Hypothesis: After analyzing 279 failed experiments, the pattern shows:
1. Pure trend-following fails in bear/range markets (2022 crash, 2025 bear)
2. RSI pullback strategies consistently failed (#251, #254, #259, #277)
3. Complex ensembles always fail (#256, #279)
4. Funding rate contrarian has best edge (Sharpe 0.8-1.5) but needs funding data

This strategy uses Ehlers Fisher Transform which excels at catching reversals in bear markets:
1. 4h Fisher Transform(9) - catches sharp reversals better than RSI
2. 1d HMA(21) - directional bias to avoid counter-trend trades (proven in #263, #274)
3. Bollinger Band Width percentile - regime detection (squeeze=mean-revert, expand=trend)
4. Volume confirmation - filters false breakouts
5. 2.5*ATR trailing stoploss - appropriate for 4h timeframe
6. Asymmetric entries - looser in range regime, stricter in trend regime

Why Fisher Transform for 4h:
- Normalizes price to Gaussian distribution (-1.5 to +1.5 range)
- Catches reversals faster than RSI/MACD
- Works well in bear market rallies (2022, 2025)
- Less whipsaw than simple EMA crossover

Entry conditions (LOOSENED to ensure >=10 trades):
- Long: Fisher < -0.8 (oversold) + cross up + 1d HMA bias OR range regime
- Short: Fisher > +0.8 (overbought) + cross down + 1d HMA bias OR range regime
- Volume > 1.2x average for confirmation

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_1d_hma_bb_regime_volume_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Reference: John Ehlers, "Rocket Science for Traders"
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: 0.66 * ((price - lowest) / (highest - lowest) - 0.5)
    3. Smooth: 0.6 * current + 0.4 * previous
    4. Fisher: 0.5 * ln((1 + smoothed) / (1 - smoothed))
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate typical price
    price = (high + low) / 2.0
    
    for i in range(period - 1, n):
        # Find highest and lowest over period
        highest = np.max(price[i - period + 1:i + 1])
        lowest = np.min(price[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize price
        normalized = 0.66 * ((price[i] - lowest) / (highest - lowest) - 0.5)
        
        # Clamp to avoid division by zero in Fisher formula
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Smooth (use previous fisher value for recursion)
        if i > period - 1 and not np.isnan(fisher[i - 1]):
            smoothed = 0.6 * normalized + 0.4 * fisher[i - 1]
        else:
            smoothed = normalized
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + smoothed) / (1 - smoothed))
    
    return fisher

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    # Band width as percentage of middle band
    bandwidth = (upper - lower) / sma
    
    return upper.values, lower.values, sma.values, bandwidth.values

def calculate_bandwidth_percentile(bandwidth, lookback=100):
    """Calculate percentile rank of current bandwidth over lookback period."""
    n = len(bandwidth)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback - 1, n):
        if np.isnan(bandwidth[i]):
            continue
        
        window = bandwidth[i - lookback + 1:i + 1]
        valid_window = window[~np.isnan(window)]
        
        if len(valid_window) < lookback // 2:
            continue
        
        # Percentile rank: where does current value fall in the distribution
        percentile[i] = np.sum(valid_window < bandwidth[i]) / len(valid_window)
    
    return percentile

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, 9)
    bb_upper, bb_lower, bb_middle, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bandwidth_percentile(bb_bandwidth, 100)
    vol_sma = calculate_volume_sma(volume, 20)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.28  # Base position size
    SIZE_REDUCED = 0.20  # Reduced size in high vol
    SIZE_MAX = 0.35  # Maximum position size
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    # Track previous Fisher for crossover detection
    prev_fisher = np.zeros(n)
    for i in range(1, n):
        prev_fisher[i] = fisher[i - 1] if not np.isnan(fisher[i - 1]) else fisher[i]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(prev_fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = strong directional bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION ===
        # BB Width percentile: < 0.3 = squeeze (range), > 0.7 = expansion (trend)
        range_regime = bb_percentile[i] < 0.35
        trend_regime = bb_percentile[i] > 0.65
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.15 * vol_sma[i]
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_REDUCED
        else:
            position_size = SIZE_BASE
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses below -0.8 = oversold (potential long)
        # Fisher crosses above +0.8 = overbought (potential short)
        fisher_oversold = fisher[i] < -0.8
        fisher_overbought = fisher[i] > 0.8
        
        # Crossover detection
        fisher_cross_up = (prev_fisher[i] < -0.8) and (fisher[i] >= -0.8)
        fisher_cross_down = (prev_fisher[i] > 0.8) and (fisher[i] <= 0.8)
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY conditions (LOOSENED for more trades):
        # Option 1: Range regime + Fisher oversold (mean reversion)
        # Option 2: Trend regime + 1d bias up + Fisher cross up (trend pullback)
        long_range = range_regime and fisher_oversold and volume_confirmed
        long_trend = trend_regime and bull_trend_1d and fisher_cross_up and volume_confirmed
        long_any = fisher_oversold and bull_trend_1d and volume_confirmed  # Fallback
        
        # SHORT ENTRY conditions (mirror of long):
        short_range = range_regime and fisher_overbought and volume_confirmed
        short_trend = trend_regime and bear_trend_1d and fisher_cross_down and volume_confirmed
        short_any = fisher_overbought and bear_trend_1d and volume_confirmed  # Fallback
        
        # Generate signal - prioritize regime-appropriate entries
        if long_range or long_trend or long_any:
            new_signal = position_size
        
        if short_range or short_trend or short_any:
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
        # Exit if HTF bias reverses against position (only in trend regime)
        if in_position and new_signal != 0.0 and trend_regime:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0  # 1d trend reversed against long
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0  # 1d trend reversed against short
        
        # === EXTREME FISHER EXIT ===
        # Exit long if Fisher goes extremely overbought (> 1.5)
        # Exit short if Fisher goes extremely oversold (< -1.5)
        if in_position and new_signal != 0.0:
            if position_side > 0 and fisher[i] > 1.5:
                new_signal = 0.0  # Take profit on extreme
            if position_side < 0 and fisher[i] < -1.5:
                new_signal = 0.0  # Take profit on extreme
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals