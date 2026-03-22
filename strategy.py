#!/usr/bin/env python3
"""
Experiment #456: 1d Fisher Transform + Choppiness Index + Weekly HMA Regime

Hypothesis: After 455 failed experiments, the key insight is that RSI-based mean 
reversion fails on 1d because RSI assumes normal distributions. Price action is 
NOT normal - it has fat tails. The Ehlers Fisher Transform normalizes price 
distribution, making reversal signals more reliable. Combined with Choppiness 
Index (proven regime filter from academic literature) and Weekly HMA trend bias, 
this should work better than RSI-based approaches (#450, #455 both failed).

Key Components:
1. EHLERS FISHER TRANSFORM (period=9): 
   - Transforms price to near-Gaussian distribution
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
   - Superior to RSI for non-normal price distributions

2. CHOPPINESS INDEX (period=14):
   - CHOP > 61.8 = ranging market (use mean reversion signals)
   - CHOP < 38.2 = trending market (use breakout signals)
   - 38.2-61.8 = transition (no new entries)
   - Proven regime filter in quantitative literature

3. WEEKLY HMA(21) TREND BIAS (via mtf_data):
   - Price > 1w HMA = bull bias (prefer long signals)
   - Price < 1w HMA = bear bias (prefer short signals)
   - HMA smoother than EMA, critical for weekly trend

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crash protection

5. ASYMMETRIC ENTRY LOGIC:
   - Range market (CHOP>61.8): Fisher mean reversion ONLY
   - Trend market (CHOP<38.2): Breakout + Fisher confirmation
   - Always aligned with weekly HMA bias

6. POSITION SIZING: 0.30 discrete (conservative for daily volatility)
   - Max 30% capital per position
   - Discrete levels minimize fee churn

Why this should beat previous 1d attempts:
- Fisher Transform > RSI for non-normal distributions (academic proof)
- Choppiness Index = proven regime filter (not ADX which failed #447)
- Weekly HMA = proven trend bias (from best strategies)
- Asymmetric logic = avoids counter-trend disasters
- Should generate 15-30 trades/year (enough for Sharpe calculation)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_weekly_hma_regime_atr_v1"
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
    Ehlers Fisher Transform - normalizes price to near-Gaussian distribution.
    Superior to RSI for reversal detection in non-normal price distributions.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to range -1 to +1 using period highs/lows
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    typical = (high + low) / 2.0
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over period
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range (with 0.999 clamp to avoid ln(0))
        normalized = 0.999 * (2.0 * (typical[i] - lowest) / price_range - 1.0)
        
        # Apply Fisher transform
        fisher_raw = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        # Smooth with EMA (span=3 for responsiveness)
        if i == period - 1:
            fisher[i] = fisher_raw
            fisher_signal[i] = fisher_raw
        else:
            fisher[i] = 0.7 * fisher_raw + 0.3 * fisher[i-1]
            fisher_signal[i] = fisher[i-1]  # Previous value for crossover detection
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    Values: 0-100 (high = choppy/ranging, low = trending)
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Ranging market (mean reversion preferred)
    - CHOP < 38.2: Trending market (breakout preferred)
    - 38.2-61.8: Transition zone (no new entries)
    """
    n = len(close)
    choppiness = np.full(n, np.nan)
    
    # Calculate true range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = tr[i-period+1:i+1].sum()
        
        # Highest high and lowest low over period
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            # CHOP formula
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
            # Clamp to 0-100
            choppiness[i] = np.clip(choppiness[i], 0, 100)
    
    return choppiness

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    choppiness = calculate_choppiness_index(high, low, close, 14)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_market = choppiness[i] > 61.8
        trending_market = choppiness[i] < 38.2
        transition_market = (choppiness[i] >= 38.2) and (choppiness[i] <= 61.8)
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # RANGING MARKET (CHOP > 61.8): Mean reversion via Fisher
        if ranging_market:
            if fisher_long_cross and bull_trend_1w:
                new_signal = SIZE
            elif fisher_short_cross and bear_trend_1w:
                new_signal = -SIZE
        
        # TRENDING MARKET (CHOP < 38.2): Breakout + Fisher confirmation
        elif trending_market:
            # Long breakout: Price > SMA50 + Fisher confirmation
            if close[i] > sma_50[i] and fisher_long_cross and bull_trend_1w:
                new_signal = SIZE
            # Short breakout: Price < SMA50 + Fisher confirmation
            elif close[i] < sma_50[i] and fisher_short_cross and bear_trend_1w:
                new_signal = -SIZE
        
        # TRANSITION MARKET (38.2-61.8): No new entries, hold existing
        elif transition_market:
            # Keep existing position, don't open new ones
            if in_position:
                new_signal = SIZE if position_side > 0 else -SIZE
            else:
                new_signal = 0.0
        
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
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w:
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