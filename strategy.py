#!/usr/bin/env python3
"""
Experiment #379: 15m Regime-Adaptive with 1h HMA Trend + RSI(7) Mean-Reversion + Supertrend

Hypothesis: 15m timeframe needs faster signals than 4h strategies, but still requires
HTF filter to avoid noise. Key insight from 378 failed experiments:

1. REGIME DETECTION is essential - trending vs ranging requires different logic
2. HTF trend bias (1h HMA) prevents counter-trend trades that get stopped out
3. RSI(7) faster than RSI(14) for 15m - catches quicker reversals
4. Supertrend for trending regime entries (breakout confirmation)
5. Asymmetric thresholds: easier to enter with HTF trend, harder against it

STRATEGY COMPONENTS:
1. 1h HMA(21): Trend bias from HTF (call get_htf_data ONCE before loop)
2. Choppiness Index(14): Regime detection on 15m
   - CHOP > 61.8 = ranging (use RSI mean-reversion)
   - CHOP < 38.2 = trending (use Supertrend breakout)
3. RSI(7): Fast mean-reversion for ranging market
   - Long: RSI < 25 + price > 1h HMA
   - Short: RSI > 75 + price < 1h HMA
4. Supertrend(10, 3.0): Trend-following for trending market
   - Long: Supertrend flips green + price > 1h HMA
   - Short: Supertrend flips red + price < 1h HMA
5. ATR(14) Trailing Stop: 2.0x ATR stoploss
6. Position Sizing: 0.25 discrete (conservative for 15m volatility)

Why this should work on 15m:
- Faster RSI(7) catches 15m reversals that RSI(14) misses
- 1h HMA provides stable trend filter without too much lag
- Regime detection avoids Supertrend whipsaw in chop
- Should generate 50-100 trades/year per symbol (enough for stats)
- Works on BTC, ETH, SOL individually (not SOL-biased)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_adaptive_1h_hma_rsi_supertrend_atr_v1"
timeframe = "15m"
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
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
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

def calculate_rsi(close, period=7):
    """Calculate RSI with specified period."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=green/long, -1=red/short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    for i in range(period, n):
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = 1  # Start with upper band (short signal)
        else:
            # Update bands based on previous direction
            if direction[i-1] == 1:  # Previously upper band
                upper_band[i] = min(upper_band[i], supertrend[i-1])
                if close[i] > supertrend[i-1]:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1  # Flip to lower band (long signal)
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
            else:  # Previously lower band
                lower_band[i] = max(lower_band[i], supertrend[i-1])
                if close[i] < supertrend[i-1]:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1  # Flip to upper band (short signal)
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 7)  # Fast RSI for 15m
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Track Supertrend direction for flip detection
    prev_st_direction = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        
        # === 1h HMA TREND BIAS ===
        bull_trend_1h = close[i] > hma_1h_aligned[i]
        bear_trend_1h = close[i] < hma_1h_aligned[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # RANGING REGIME: RSI Mean-Reversion
        if ranging_market:
            # Long: RSI oversold + with HTF trend (or neutral)
            if rsi[i] < 25 and bull_trend_1h:
                new_signal = SIZE
            # Short: RSI overbought + with HTF trend (or neutral)
            elif rsi[i] > 75 and bear_trend_1h:
                new_signal = -SIZE
        
        # TRENDING REGIME: Supertrend Breakout
        elif trending_market:
            # Detect Supertrend flip
            st_flip_long = prev_st_direction == 1 and st_direction[i] == -1
            st_flip_short = prev_st_direction == -1 and st_direction[i] == 1
            
            # Long: Supertrend flips green + with HTF trend
            if st_flip_long and bull_trend_1h:
                new_signal = SIZE
            # Short: Supertrend flips red + with HTF trend
            elif st_flip_short and bear_trend_1h:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if HTF trend turns bear
        if in_position and position_side > 0 and bear_trend_1h:
            new_signal = 0.0
        
        # Exit short if HTF trend turns bull
        if in_position and position_side < 0 and bull_trend_1h:
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
        
        # Update previous Supertrend direction for next iteration
        prev_st_direction = st_direction[i] if not np.isnan(st_direction[i]) else prev_st_direction
        
        signals[i] = new_signal
    
    return signals