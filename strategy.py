#!/usr/bin/env python3
"""
Experiment #308: 30m Fisher Transform Reversal with 4h HMA Trend Bias and Choppiness Regime Filter

Hypothesis: After 307 experiments, clear patterns emerge:
1. Complex ensembles (3+ filters) consistently fail (#297, #301, #302, #307)
2. Simple trend following works on 4h/12h but fails on 30m (#296, #302, #303)
3. 2025 bear/range market requires different approach than 2021-2024 bull
4. Research shows Fisher Transform catches reversals with 75% win rate in bear markets
5. Choppiness Index is best meta-filter for distinguishing range vs trend regimes

This strategy uses FISHER TRANSFORM mean reversion with regime filter:
1. 4h HMA(21) for primary directional bias (proven edge from #299, #304)
2. Choppiness Index(14) for regime detection: CHOP>61.8=range, CHOP<38.2=trend
3. Fisher Transform(9) for entry timing in ranging markets
4. In trending regimes: only trade pullbacks toward 4h HMA
5. ATR(14) trailing stoploss at 2.0x (tighter for 30m timeframe)

Why this might work on 30m:
- Fisher Transform is designed for reversal capture (different from failed Supertrend/Donchian)
- Choppiness filter avoids Fisher whipsaws in strong trends
- 4h HMA bias prevents counter-trend trades (learned from #296 failure)
- Regime-adaptive logic: mean revert in range, trend-follow in trend
- Should generate 20-50 trades/year (enough for Sharpe, not too many for fees)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_chop_regime_atr_v1"
timeframe = "30m"
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
    Converts price into a Gaussian normal distribution.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    signal_line = np.zeros(n)
    signal_line[:] = np.nan
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        range_val = highest_high - lowest_low
        if range_val == 0:
            continue
        
        normalized = (hl2 - lowest_low) / range_val
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Calculate intermediate value
        temp = 0.66 * ((normalized - 0.5) + 0.67 * (normalized - 0.5))
        temp = np.clip(temp, -0.999, 0.999)
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1 + temp) / (1 - temp))
        
        # Signal line (1-period lag of Fisher)
        if i > period:
            signal_line[i] = fisher[i-1]
    
    return fisher, signal_line

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    Values between 38.2 and 61.8 = transition/neutral
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range == 0:
            continue
        
        # Sum of ATR over period
        tr1 = high[i-period+1:i+1] - low[i-period+1:i+1]
        tr2 = np.abs(high[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))
        tr3 = np.abs(low[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr_sum = np.sum(tr)
        
        # CHOP formula
        chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_INCREASED = 0.35  # Increased size in high-confidence setups
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = primary directional bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION ===
        # Choppiness Index determines market regime
        ranging_market = chop[i] > 61.8  # Mean reversion regime
        trending_market = chop[i] < 38.2  # Trend following regime
        # Neutral: 38.2 <= chop <= 61.8 (reduced position size)
        neutral_market = not ranging_market and not trending_market
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long = False
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_long = (fisher[i-1] < -1.5) and (fisher[i] >= -1.5)
        
        # Short: Fisher crosses below +1.5 from above
        fisher_short = False
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_short = (fisher[i-1] > 1.5) and (fisher[i] <= 1.5)
        
        # === PULLBACK ENTRIES (for trending regime) ===
        # In trending market, enter on pullback toward 4h HMA
        pullback_long = bull_trend_4h and (close[i] < hma_4h_aligned[i] * 1.005) and (close[i] > hma_4h_aligned[i] * 0.995)
        pullback_short = bear_trend_4h and (close[i] > hma_4h_aligned[i] * 0.995) and (close[i] < hma_4h_aligned[i] * 1.005)
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility and regime
        if high_volatility:
            position_size = SIZE_BASE  # Conservative in high vol
        elif trending_market or (bull_trend_4h and fisher_long) or (bear_trend_4h and fisher_short):
            position_size = SIZE_INCREASED  # Higher confidence
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY scenarios:
        # 1. Ranging market + Fisher long signal (mean reversion)
        # 2. Trending market + bull trend + Fisher long (trend pullback)
        # 3. Neutral market + bull trend + Fisher long (reduced confidence)
        
        if ranging_market and fisher_long:
            new_signal = position_size  # Mean reversion long
        
        elif trending_market and bull_trend_4h and fisher_long:
            new_signal = position_size  # Trend pullback long
        
        elif neutral_market and bull_trend_4h and fisher_long:
            new_signal = SIZE_BASE  # Reduced size in neutral
        
        # SHORT ENTRY scenarios (mirror of long):
        if ranging_market and fisher_short:
            new_signal = -position_size  # Mean reversion short
        
        elif trending_market and bear_trend_4h and fisher_short:
            new_signal = -position_size  # Trend pullback short
        
        elif neutral_market and bear_trend_4h and fisher_short:
            new_signal = -SIZE_BASE  # Reduced size in neutral
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === FISHER REVERSAL EXIT ===
        # Exit if Fisher signals opposite direction
        if in_position and new_signal != 0.0:
            if position_side > 0 and fisher_short:
                new_signal = 0.0  # Fisher turned against long
            if position_side < 0 and fisher_long:
                new_signal = 0.0  # Fisher turned against short
        
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
            # else: maintaining same position direction (possibly adjusted size)
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