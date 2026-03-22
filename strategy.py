#!/usr/bin/env python3
"""
Experiment #417: 1h Choppiness Index Regime + 4h HMA Trend + RSI/Fisher Adaptive Entry

Hypothesis: After 416 experiments, the key insight is that REGIME DETECTION is critical.
BTC/ETH spend ~60% of time in ranging markets where trend strategies fail. This strategy:

1. CHOPPINESS INDEX (CHOP) REGIME on 1h:
   - CHOP > 61.8 = ranging (use mean-reversion entries)
   - CHOP < 38.2 = trending (use momentum entries)
   - 38.2-61.8 = neutral (stay flat or reduce position)
   - This meta-filter prevents trend strategies from dying in ranges

2. 4h HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 4h HMA
   - Short bias when price < 4h HMA
   - HMA smoother than EMA, critical for MTF alignment

3. ADAPTIVE ENTRY LOGIC:
   - Ranging regime: RSI(14) extremes (30/70) for mean-reversion
   - Trending regime: Fisher Transform(9) crosses for momentum
   - Both filtered by 4h HMA trend bias (no counter-trend)

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from 2022-style crashes

5. POSITION SIZING: 0.25 discrete (conservative for 1h volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why 1h with this approach:
- Faster reaction than 4h/12h strategies
- Choppiness filter prevents whipsaw in ranges
- Fisher Transform catches reversals better than RSI in trends
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_4h_hma_fisher_rsi_adaptive_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Catches reversals in bear/bull markets better than RSI.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1].max() + low[i-period+1:i+1].min()) / 2.0
        
        # Normalize to -1 to +1 range
        highest_hl = high[i-period+1:i+1].max()
        lowest_hl = low[i-period+1:i+1].min()
        
        if highest_hl - lowest_hl > 1e-10:
            normalized = 0.66 * ((hl2 - lowest_hl) / (highest_hl - lowest_hl) - 0.5)
            normalized = max(-0.99, min(0.99, normalized))
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            if i > period:
                fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
            
            # Trigger line (1-period lag)
            trigger[i] = fisher[i-1] if i > period else fisher[i]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean-reversion)
    CHOP < 38.2 = trending market (momentum)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high - lowest_low > 1e-10:
            # Sum of ATR over period
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr1 = high[j] - low[j]
                tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
                tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
                tr_sum += max(tr1, tr2, tr3)
            
            # CHOP formula
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        # neutral_market = 38.2 <= CHOP <= 61.8 (reduce position or stay flat)
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # RANGING REGIME: RSI mean-reversion with 4h HMA filter
        if ranging_market:
            if bull_trend_4h and rsi[i] < 35:
                new_signal = SIZE
            elif bear_trend_4h and rsi[i] > 65:
                new_signal = -SIZE
        
        # TRENDING REGIME: Fisher Transform momentum with 4h HMA filter
        elif trending_market:
            # Fisher cross above -1.5 = long signal
            if bull_trend_4h and fisher[i] > -1.5 and fisher_trigger[i] <= -1.5:
                new_signal = SIZE
            # Fisher cross below +1.5 = short signal
            elif bear_trend_4h and fisher[i] < 1.5 and fisher_trigger[i] >= 1.5:
                new_signal = -SIZE
        
        # NEUTRAL REGIME: Stay flat (38.2 <= CHOP <= 61.8)
        # This avoids whipsaw when regime is unclear
        
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
            # Long position entered in ranging should exit if market becomes trending without Fisher signal
            if position_side > 0 and trending_market:
                # Keep position if Fisher still bullish
                if fisher[i] < -1.0:
                    new_signal = 0.0
            # Short position entered in trending should exit if market becomes ranging without RSI signal
            if position_side < 0 and ranging_market:
                # Keep position if RSI still bearish
                if rsi[i] < 60:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
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