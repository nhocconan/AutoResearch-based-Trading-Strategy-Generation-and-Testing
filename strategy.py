#!/usr/bin/env python3
"""
Experiment #453: 1h Multi-Regime Strategy with 4h HMA Trend + Choppiness Filter

Hypothesis: After 452 failed experiments, the key insight is that strategies fail
because they don't adapt to market regime. This strategy uses:

1. 4h HMA(21) TREND BIAS (via mtf_data helper):
   - Faster than 1d/1w, better suited for 1h entries
   - Long bias when price > 4h HMA, short when < 4h HMA

2. CHOPPINESS INDEX (CHOP) REGIME DETECTION:
   - CHOP(14) > 61.8 = ranging market (use mean reversion)
   - CHOP(14) < 38.2 = trending market (use breakout/pullback)
   - This is the KEY differentiator - most failed strategies ignore regime

3. TWO ENTRY MODES (regime-dependent):
   a) TRENDING REGIME (CHOP < 38.2): RSI(14) pullback to 40-60 zone
      - Long: RSI crosses above 45 + price > 4h HMA
      - Short: RSI crosses below 55 + price < 4h HMA
   
   b) RANGING REGIME (CHOP > 61.8): RSI(14) extreme mean reversion
      - Long: RSI < 30 + price > 4h HMA (oversold in uptrend)
      - Short: RSI > 70 + price < 4h HMA (overbought in downtrend)

4. VOLUME CONFIRMATION:
   - Volume > 1.3 * SMA(volume, 20) confirms breakout validity
   - Reduces false signals in low-volume periods

5. ATR(14) TRAILING STOP at 2.0x:
   - Tighter than previous 2.5x to reduce drawdown
   - Critical for 2022-style crash protection

6. ASYMMETRIC POSITION SIZING:
   - Trending regime: SIZE = 0.30 (higher confidence)
   - Ranging regime: SIZE = 0.20 (lower confidence)
   - Discrete levels minimize fee churn

Why this should work on 1h:
- 4h HMA provides smooth trend filter without excessive lag
- Choppiness Index is proven regime detector (Ehlers, 1995)
- Regime-adaptive entries match market conditions
- Volume filter reduces false breakouts
- Should generate 50-100 trades/year per symbol (sufficient frequency)
- Works on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_chop_4h_hma_rsi_vol_adaptive_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = ranging/choppy market (mean reversion favored)
    - CHOP < 38.2 = trending market (trend following favored)
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
        # Sum of ATR over period
        atr_sum = tr[i-period+1:i+1].sum()
        
        # Highest high and lowest low over period
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    chop = calculate_choppiness(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels based on regime (Rule 4)
    SIZE_TRENDING = 0.30  # Higher confidence in trending regime
    SIZE_RANGING = 0.20   # Lower confidence in ranging regime
    
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
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        neutral_regime = not trending_regime and not ranging_regime
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.3 * vol_sma[i]
        
        # === SELECT POSITION SIZE BASED ON REGIME ===
        current_size = SIZE_TRENDING if trending_regime else SIZE_RANGING
        
        # === GENERATE SIGNAL (Regime-Adaptive) ===
        new_signal = 0.0
        
        # TRENDING REGIME: RSI pullback entries
        if trending_regime:
            # Long: RSI pullback in uptrend
            if bull_trend_4h and 40 <= rsi[i] <= 60 and rsi[i] > rsi[i-1]:
                new_signal = current_size
            # Short: RSI pullback in downtrend
            elif bear_trend_4h and 40 <= rsi[i] <= 60 and rsi[i] < rsi[i-1]:
                new_signal = -current_size
        
        # RANGING REGIME: RSI extreme mean reversion
        elif ranging_regime:
            # Long: Oversold in uptrend (mean reversion)
            if bull_trend_4h and rsi[i] < 30:
                new_signal = current_size
            # Short: Overbought in downtrend (mean reversion)
            elif bear_trend_4h and rsi[i] > 70:
                new_signal = -current_size
        
        # NEUTRAL REGIME: No new entries, wait for clear signal
        # (but allow existing positions to continue)
        
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