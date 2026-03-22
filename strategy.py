#!/usr/bin/env python3
"""
Experiment #014: 4h Choppiness Regime + KAMA Trend + 12h HMA Filter

Hypothesis: Previous strategies failed due to overly complex entry conditions that never
all aligned (resulting in 0 trades). This strategy uses a simpler regime-switching approach:

1. Choppiness Index (CHOP 14) - Regime detection
   - CHOP > 61.8 = Range-bound market → Mean reversion logic
   - CHOP < 38.2 = Trending market → Trend following logic
   - Between = Neutral → Reduce position size or stay flat

2. KAMA (Kaufman Adaptive Moving Average) - Adaptive trend indicator
   - Adapts speed based on market efficiency ratio
   - Less whipsaw than EMA during choppy periods
   - Proven in crypto markets (ETH Sharpe +0.755 in similar config)

3. 12h HMA(21) - Higher timeframe trend filter via mtf_data helper
   - Only long if price > 12h HMA
   - Only short if price < 12h HMA
   - Prevents counter-trend trades against major direction

4. Simple Entry Scoring - Ensures trade frequency
   - Each condition adds points, entry at threshold >= 4
   - Avoids the "all conditions must align" problem that caused 0 trades

5. ATR(14) Trailing Stop - 2.5x ATR for risk management
   - Signal → 0 when stopped out

Why this should beat current best (Sharpe=0.335):
- Regime switching adapts to market conditions (bull/bear/range)
- KAMA adapts to volatility, reducing whipsaw vs fixed EMA
- 12h HTF filter prevents major counter-trend losses
- Simpler entry logic = more trades (avoids 0-trade failure)
- 4h timeframe targets 20-50 trades/year (optimal fee/return balance)

Timeframe: 4h (REQUIRED for Experiment #014)
HTF: 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 high conviction, 0.15 low conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_kama_regime_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - Measures market trendiness vs choppiness.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8 = Range-bound / Choppy market (mean reversion favored)
    - CHOP < 38.2 = Trending market (trend following favored)
    - 38.2 - 61.8 = Transition / Neutral
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    price_range = highest_high - lowest_low
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.nan)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing constant based on market efficiency ratio.
    
    Efficiency Ratio (ER) = |Price Change| / Sum of |Individual Price Changes|
    - ER near 1 = Strong trend (use fast smoothing)
    - ER near 0 = Choppy/noise (use slow smoothing)
    
    Smoothing Constant (SC) = [ER * (fast_SC - slow_SC) + slow_SC]^2
    KAMA = Previous KAMA + SC * (Price - Previous KAMA)
    """
    close_s = pd.Series(close)
    n = len(close_s)
    
    # Price change over period
    price_change = np.abs(close_s - close_s.shift(period))
    
    # Sum of individual price changes
    individual_changes = np.abs(close_s.diff())
    sum_changes = individual_changes.rolling(window=period, min_periods=period).sum()
    
    # Efficiency Ratio
    er = price_change / sum_changes.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA iteratively
    kama = np.zeros(n)
    kama[0] = close_s.iloc[0]
    
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper.values, lower.values, middle.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend filter
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    chop_14 = calculate_choppiness(high, low, close, period=14)
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    LOW_CONV_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(kama_10[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = Range-bound (mean reversion)
        # CHOP < 38.2 = Trending (trend following)
        # 38.2 - 61.8 = Neutral/Transition
        is_ranging = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        is_neutral = not is_ranging and not is_trending
        
        # === 12H TREND FILTER ===
        trend_12h_bullish = close[i] > hma_12h_21_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # KAMA slope (trend strength)
        kama_slope_bullish = False
        kama_slope_bearish = False
        if i > 5:
            kama_slope_bullish = kama_10[i] > kama_10[i-5]
            kama_slope_bearish = kama_10[i] < kama_10[i-5]
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5
        near_bb_lower = bb_position < 0.2
        near_bb_upper = bb_position > 0.8
        
        # === ENTRY LOGIC - REGIME SWITCHING ===
        new_signal = 0.0
        
        if is_trending:
            # TREND FOLLOWING MODE
            # Long entry scoring
            long_score = 0
            
            if kama_bullish:
                long_score += 2
            if kama_slope_bullish:
                long_score += 1
            if trend_12h_bullish:
                long_score += 2
            if rsi_bullish:
                long_score += 1
            
            # Enter long if score >= 4
            if long_score >= 4:
                if trend_12h_bullish and kama_bullish:
                    new_signal = HIGH_CONV_SIZE
                else:
                    new_signal = BASE_SIZE
            
            # Short entry scoring
            short_score = 0
            
            if kama_bearish:
                short_score += 2
            if kama_slope_bearish:
                short_score += 1
            if trend_12h_bearish:
                short_score += 2
            if rsi_bearish:
                short_score += 1
            
            # Enter short if score >= 4
            if short_score >= 4:
                if trend_12h_bearish and kama_bearish:
                    new_signal = -HIGH_CONV_SIZE
                else:
                    new_signal = -BASE_SIZE
        
        elif is_ranging:
            # MEAN REVERSION MODE
            # Long when oversold + near BB lower + 12h trend not strongly bearish
            if rsi_oversold and near_bb_lower:
                if not trend_12h_bearish or chop_14[i] > 70:  # Very choppy = ignore 12h trend
                    new_signal = BASE_SIZE
            
            # Short when overbought + near BB upper + 12h trend not strongly bullish
            if rsi_overbought and near_bb_upper:
                if not trend_12h_bullish or chop_14[i] > 70:  # Very choppy = ignore 12h trend
                    new_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME - REDUCED ACTIVITY ===
        if is_neutral and new_signal == 0.0:
            # Only enter with very high conviction in neutral regime
            if kama_bullish and trend_12h_bullish and rsi_bullish:
                new_signal = LOW_CONV_SIZE
            elif kama_bearish and trend_12h_bearish and rsi_bearish:
                new_signal = -LOW_CONV_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~240 hours = 10 days on 4h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if kama_bullish and trend_12h_bullish:
                new_signal = LOW_CONV_SIZE
            elif kama_bearish and trend_12h_bearish:
                new_signal = -LOW_CONV_SIZE
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h trend turns bearish
            if position_side > 0 and trend_12h_bearish:
                trend_reversal = True
            # Exit short if 12h trend turns bullish
            if position_side < 0 and trend_12h_bullish:
                trend_reversal = True
        
        # === REGIME CHANGE EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit trend position if regime becomes highly choppy
            if chop_14[i] > 70 and position_side != 0:
                regime_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or regime_exit:
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals