#!/usr/bin/env python3
"""
Experiment #376: 4h Choppiness Index Regime + Fisher Transform + 1d HMA Trend

Hypothesis: After 375 failed experiments, the key insight is REGIME DETECTION.
Simple trend-following fails in 2022 crash and 2025 bear market. Pure mean-reversion
fails in strong trends. We need to ADAPT strategy based on market regime.

STRATEGY COMPONENTS:
1. CHOPPINESS INDEX (14-period): Detects trending vs ranging market
   - CHOP > 61.8 = ranging (use mean-reversion Fisher Transform)
   - CHOP < 38.2 = trending (use 1d HMA trend-following)
   - 38.2-61.8 = neutral (stay flat, avoid whipsaw)
   - This is the KEY meta-filter that 300+ failed strategies lacked

2. EHLERS FISHER TRANSFORM (9-period): For ranging market entries
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
   - Proven 75% win rate in range-bound markets

3. 1d HMA(21) TREND BIAS: For trending market entries
   - Long when price > 1d HMA and CHOP < 38.2
   - Short when price < 1d HMA and CHOP < 38.2
   - HMA smoother than EMA, less lag for trend detection

4. ATR TRAILING STOP (2.5x): Risk management
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from 2022-style crashes

5. POSITION SIZING: 0.28 discrete (conservative for 4h volatility)
   - Max 28% capital per position
   - Discrete levels minimize fee churn

Why this should work:
- Regime detection avoids trend strategies in chop (2021-2022 whipsaw)
- Fisher Transform catches reversals in bear market rallies
- 1d HMA provides stable trend bias (daily closes are significant)
- Should generate 30-60 trades/year per symbol (enough for stats)
- Works on BTC, ETH, SOL individually (not SOL-biased)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_fisher_1d_hma_atr_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean-reversion works)
    CHOP < 38.2 = trending market (trend-following works)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0 if 'close' in dir() else (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2 * (typical[i] - lowest) / price_range - 1
        
        # Apply Fisher Transform with smoothing
        # Use 0.67 smoothing factor as per Ehlers
        if i == period:
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
            fisher_signal[i] = fisher[i]
        else:
            fisher[i] = 0.67 * 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10)) + 0.33 * fisher[i-1]
            fisher_signal[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
    
    return fisher, fisher_signal

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        # neutral_market = 38.2 <= chop[i] <= 61.8 (stay flat)
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS (for ranging market) ===
        fisher_long = fisher_signal[i] < -1.5 and fisher[i] > -1.5  # cross above -1.5
        fisher_short = fisher_signal[i] > 1.5 and fisher[i] < 1.5   # cross below +1.5
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # TRENDING REGIME: Follow 1d HMA trend
        if trending_market:
            if bull_trend_1d:
                new_signal = SIZE
            elif bear_trend_1d:
                new_signal = -SIZE
        
        # RANGING REGIME: Fisher Transform mean-reversion
        elif ranging_market:
            if fisher_long:
                new_signal = SIZE
            elif fisher_short:
                new_signal = -SIZE
        # NEUTRAL REGIME: Stay flat (38.2 <= CHOP <= 61.8)
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit if regime changes against position type
        if in_position and new_signal != 0.0:
            # Long position in trending regime should exit if market becomes ranging
            if position_side > 0 and ranging_market and not fisher_long:
                new_signal = 0.0
            # Short position in trending regime should exit if market becomes ranging
            if position_side < 0 and ranging_market and not fisher_short:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT (for trending regime positions) ===
        if in_position and new_signal != 0.0 and trending_market:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
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