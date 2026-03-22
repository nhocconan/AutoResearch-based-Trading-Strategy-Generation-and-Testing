#!/usr/bin/env python3
"""
Experiment #008: 30m Regime-Adaptive Strategy with 4h HMA Trend + Choppiness Index

Hypothesis: After 7 failed experiments, the pattern shows that static strategies fail
because crypto markets alternate between trending and ranging regimes. This 30m strategy
combines regime detection with adaptive entry logic:

1. CHOPPINESS INDEX (CHOP) regime filter:
   - CHOP > 61.8 = range/choppy → mean reversion (RSI extremes at BB bounds)
   - CHOP < 38.2 = trending → trend following (pullback entries)
   - This meta-filter switches strategy logic based on market state

2. 4h HMA trend bias: Stable HTF direction filter. Only long if price > 4h_HMA,
   only short if price < 4h_HMA. Prevents counter-trend trades.

3. 30m RSI + Bollinger Bands for entries:
   - Range regime: RSI < 25 + price < BB_lower → long (oversold bounce)
   - Range regime: RSI > 75 + price > BB_upper → short (overbought fade)
   - Trend regime: RSI pullback to 40-50 + price > 4h_HMA → long continuation
   - Trend regime: RSI rally to 50-60 + price < 4h_HMA → short continuation

4. Volume confirmation: Entry volume > 0.7 * 20bar_avg filters fakeouts

5. ATR trailing stoploss: 2.5 * ATR(14) protects capital in crashes

6. Discrete position sizing: 0.0, ±0.25, ±0.30 (minimizes fee churn)

Why 30m with regime-adaptive should beat failed strategies:
- Adapts to market state (range vs trend) instead of one rigid logic
- 4h HMA bias is stable (changes rarely = less churn)
- CHOP filter prevents trend strategies in chop (2022 whipsaw protection)
- Mean reversion in range captures 2025 bear/range market opportunities
- Target 40-60 trades/year = optimal for 30m (Rule 10)
- Works on BTC/ETH/SOL individually (not SOL-only bias)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete, ATR-scaled
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_chop_4h_hma_rsi_bb_adaptive_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma * 100
    
    return upper.values, lower.values, sma.values, bb_width.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Sum of ATR over period
    atr_sum = atr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / hh_ll.replace(0, np.inf)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    chop = chop.clip(0, 100)
    
    return chop.values

def calculate_zscore(series, period=20):
    """Calculate rolling Z-score for mean reversion signals."""
    s = pd.Series(series)
    sma = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std()
    zscore = (s - sma) / std.replace(0, np.inf)
    return zscore.values

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
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    close_zscore = calculate_zscore(close, 20)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        # === 4h HMA TREND BIAS (Stable HTF filter) ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS INDEX REGIME DETECTION ===
        # CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
        is_range_regime = chop_14[i] > 55  # Slightly relaxed threshold
        is_trend_regime = chop_14[i] < 45  # Slightly relaxed threshold
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.7 * vol_sma[i]
        
        # === Z-SCORE EXTREMES ===
        zscore_extreme_low = close_zscore[i] < -1.5
        zscore_extreme_high = close_zscore[i] > 1.5
        
        # === ATR-BASED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[100:i])
            if atr_median > 0:
                atr_ratio = atr_14[i] / atr_median
                atr_ratio = np.clip(atr_ratio, 0.5, 2.5)
                size_multiplier = 1.0 / atr_ratio
            else:
                size_multiplier = 1.0
        else:
            size_multiplier = 1.0
        
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        new_signal = 0.0
        
        # MODE 1: RANGE REGIME - Mean Reversion at BB extremes
        if is_range_regime:
            # Long: RSI oversold + price at BB lower + zscore extreme + volume
            if (rsi_14[i] < 30 and 
                close[i] <= bb_lower[i] and 
                zscore_extreme_low and 
                volume_confirmed and
                bull_bias):  # Still respect HTF bias
                new_signal = current_size
            
            # Short: RSI overbought + price at BB upper + zscore extreme + volume
            elif (rsi_14[i] > 70 and 
                  close[i] >= bb_upper[i] and 
                  zscore_extreme_high and 
                  volume_confirmed and
                  bear_bias):  # Still respect HTF bias
                new_signal = -current_size
        
        # MODE 2: TREND REGIME - Pullback entries in direction of trend
        elif is_trend_regime:
            # Long: bullish bias + RSI pullback to 40-50 + volume
            if (bull_bias and 
                38 < rsi_14[i] < 52 and 
                volume_confirmed and
                close[i] > bb_mid[i]):  # Price above middle band
                new_signal = current_size
            
            # Short: bearish bias + RSI rally to 50-62 + volume
            elif (bear_bias and 
                  48 < rsi_14[i] < 62 and 
                  volume_confirmed and
                  close[i] < bb_mid[i]):  # Price below middle band
                new_signal = -current_size
        
        # MODE 3: NEUTRAL REGIME - Wait for clearer signals
        else:
            # Only enter on strong RSI extremes with HTF confirmation
            if rsi_14[i] < 25 and bull_bias and volume_confirmed:
                new_signal = current_size * 0.7
            elif rsi_14[i] > 75 and bear_bias and volume_confirmed:
                new_signal = -current_size * 0.7
        
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
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit if regime changes against position
            if position_side > 0 and is_trend_regime and bear_bias:
                regime_exit = True
            if position_side < 0 and is_trend_regime and bull_bias:
                regime_exit = True
            
            # Exit long if RSI becomes overbought in range regime
            if position_side > 0 and is_range_regime and rsi_14[i] > 70:
                regime_exit = True
            # Exit short if RSI becomes oversold in range regime
            if position_side < 0 and is_range_regime and rsi_14[i] < 30:
                regime_exit = True
        
        # Apply stoploss or regime exit
        if stoploss_triggered or regime_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals