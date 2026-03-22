#!/usr/bin/env python3
"""
Experiment #326: 30m Regime-Adaptive Strategy with 4h HMA Bias

Hypothesis: After #320 failed (Sharpe=-1.028) with simple EMA crossover + Choppiness,
the issue was conflicting signals between trend and mean-reversion logic.

Analysis of #316 (Sharpe=0.676, best so far) shows regime detection works on 4h.
For 30m, I'll use a CLEANER regime-adaptive approach:

1. 4h HMA(21) for PRIMARY trend bias (proven edge from multiple strategies)
2. Choppiness Index(14) to detect regime: CHOP>61.8=range, CHOP<38.2=trend
3. RANGE regime (CHOP>61.8): Mean revert at Bollinger Band extremes + RSI confirmation
4. TREND regime (CHOP<38.2): Follow 4h HMA direction with RSI pullback entries
5. NEUTRAL regime (38.2<=CHOP<=61.8): Stay flat or reduce position size

Key differences from #320:
- Simpler regime logic (no conflicting filters)
- RSI pullback in trend regime (not crossover)
- BB mean reversion ONLY in range regime
- Clear separation between regime behaviors

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adaptive_4h_hma_bb_rsi_atr_v1"
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
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n) * np.nan
    
    for i in range(period, n):
        # Calculate ATR for each bar in the lookback period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j - 1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j - 1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_RANGE = 0.25    # Mean reversion in range regime
    SIZE_TREND = 0.30    # Trend following in trend regime
    SIZE_WEAK = 0.15     # Reduced size in neutral regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        in_range_regime = chop[i] > 61.8
        in_trend_regime = chop[i] < 38.2
        in_neutral_regime = not in_range_regime and not in_trend_regime
        
        # === 4H TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 35 <= rsi[i] <= 65
        
        # === BOLLINGER BAND CONDITIONS ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # At or below lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # At or above upper band
        near_bb_mid = bb_lower[i] * 1.01 < close[i] < bb_upper[i] * 0.99
        
        # === DETERMINE POSITION SIZE BASED ON REGIME ===
        if in_trend_regime:
            position_size = SIZE_TREND
        elif in_range_regime:
            position_size = SIZE_RANGE
        else:
            position_size = SIZE_WEAK
        
        # === GENERATE SIGNAL BASED ON REGIME ===
        new_signal = 0.0
        
        if in_range_regime:
            # RANGE REGIME: Mean revert at BB extremes
            # Long when price at lower BB + RSI oversold
            if at_bb_lower and rsi_oversold:
                new_signal = position_size
            # Short when price at upper BB + RSI overbought
            elif at_bb_upper and rsi_overbought:
                new_signal = -position_size
        
        elif in_trend_regime:
            # TREND REGIME: Follow 4h HMA direction with RSI pullback
            # Long when 4h trend up + RSI pullback (not oversold, just dipping)
            if bull_trend_4h and rsi[i] < 50 and rsi[i] > 30:
                new_signal = position_size
            # Short when 4h trend down + RSI pullback
            elif bear_trend_4h and rsi[i] > 50 and rsi[i] < 70:
                new_signal = -position_size
        
        else:
            # NEUTRAL REGIME: Only take strong signals, reduced size
            if at_bb_lower and rsi_oversold and bear_trend_4h == False:
                new_signal = SIZE_WEAK
            elif at_bb_upper and rsi_overbought and bull_trend_4h == False:
                new_signal = -SIZE_WEAK
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and in_trend_regime and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and in_trend_regime and bull_trend_4h:
                new_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 75:
                new_signal = 0.0  # Take profit on long
            if position_side < 0 and rsi[i] < 25:
                new_signal = 0.0  # Take profit on short
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position reversal
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals