#!/usr/bin/env python3
"""
Experiment #012: 12h Adaptive Regime Strategy with Fisher Transform + Choppiness Index

Hypothesis: Previous 12h strategies failed because they used单一 logic (either pure trend or pure mean-reversion).
Crypto markets switch between trending and ranging regimes frequently. This strategy adapts:

1. CHOPPINESS INDEX (14) - Regime detector:
   - CHOP > 61.8 = Range/Chop (use mean-reversion logic)
   - CHOP < 38.2 = Trend (use breakout logic)
   - Between = Neutral (reduce position size)

2. EHLERS FISHER TRANSFORM (9) - Reversal indicator for range markets.
   Long when Fisher crosses above -1.5, Short when crosses below +1.5.
   Proven to catch bear market rallies better than RSI.

3. DONCHIAN(20) BREAKOUT - Trend entry for trending regimes.
   Long on 20-bar high breakout, Short on 20-bar low breakout.

4. 1d HMA(21) + 1w HMA(21) - HTF trend filter via mtf_data helper.
   Only long if price > 1d HMA, only short if price < 1d HMA.
   Increase size when 1w aligns with 1d.

5. ATR(14) Trailing Stop - 2.5x ATR for risk management.

6. LENIENT ENTRY CONDITIONS - Ensure minimum trade frequency (critical lesson from 11 failures).
   Multiple entry paths: Fisher reversal OR Donchian breakout OR pullback to HMA.

Timeframe: 12h (REQUIRED for Experiment #012)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Target: 20-50 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_chop_regime_donchian_1d_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    Formula: 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) * 100 / log10(n)
    CHOP > 61.8 = Range, CHOP < 38.2 = Trend
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
    
    # Highest High - Lowest Low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    hh_ll = hh - ll
    
    # Choppiness Index
    chop = 100 * (atr_sum / hh_ll.replace(0, np.nan)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals better than RSI in bear markets.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high_s + low_s + close) / 3
    
    # Normalize to -1 to +1
    hh = typical.rolling(window=period, min_periods=period).max()
    ll = typical.rolling(window=period, min_periods=period).min()
    
    normalized = 2 * (typical - ll) / (hh - ll).replace(0, np.nan) - 1
    normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Apply Fisher transform
    fisher_input = normalized.replace([-1, 1], [-0.999, 0.999])  # Avoid log(0)
    fisher = 0.5 * np.log((1 + fisher_input) / (1 - fisher_input).replace(0, np.nan))
    fisher = fisher.replace([np.inf, -np.inf], np.nan)
    
    # Signal line (1-bar lag)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel highs and lows."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    donchian_high = high_s.rolling(window=period, min_periods=period).max().values
    donchian_low = low_s.rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    return donchian_high, donchian_low, donchian_mid

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend filter
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1w HMA for major bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Also calculate 12h HMA for pullback entries
    hma_12h_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.35
    LOW_CONV_SIZE = 0.20
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(fisher[i]) or np.isnan(donchian_high[i]):
            continue
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = Range (mean-revert), CHOP < 38.2 = Trend (breakout)
        is_range_regime = chop[i] > 61.8
        is_trend_regime = chop[i] < 38.2
        is_neutral_regime = not is_range_regime and not is_trend_regime
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS (for range regime) ===
        fisher_long = False
        fisher_short = False
        
        if i > 0 and not np.isnan(fisher_signal[i-1]):
            # Long: Fisher crosses above -1.5 from below
            if fisher[i] > -1.5 and fisher_signal[i-1] <= -1.5:
                fisher_long = True
            # Short: Fisher crosses below +1.5 from above
            if fisher[i] < 1.5 and fisher_signal[i-1] >= 1.5:
                fisher_short = True
        
        # === DONCHIAN BREAKOUT (for trend regime) ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        
        if i > 0:
            if close[i] > donchian_high[i-1]:
                donchian_breakout_long = True
            if close[i] < donchian_low[i-1]:
                donchian_breakout_short = True
        
        # === HMA PULLBACK ENTRY (works in both regimes) ===
        hma_pullback_long = False
        hma_pullback_short = False
        
        if i > 0:
            # Long: price pulls back to HMA in bullish trend
            if daily_bullish and close[i] <= hma_12h_21[i] * 1.005 and close[i-1] > hma_12h_21[i-1]:
                hma_pullback_long = True
            # Short: price rallies to HMA in bearish trend
            if daily_bearish and close[i] >= hma_12h_21[i] * 0.995 and close[i-1] < hma_12h_21[i-1]:
                hma_pullback_short = True
        
        # === RSI EXTREMES (additional mean-reversion signal) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC - MULTIPLE PATHS TO ENSURE TRADES ===
        new_signal = 0.0
        entry_reason = ""
        
        # LONG ENTRY - Multiple paths (LENIENT for trade frequency)
        long_score = 0
        
        # Path 1: Fisher reversal in range regime
        if fisher_long and is_range_regime:
            long_score += 3
            entry_reason = "fisher_range"
        
        # Path 2: Donchian breakout in trend regime
        if donchian_breakout_long and is_trend_regime:
            long_score += 3
            entry_reason = "donchian_trend"
        
        # Path 3: HMA pullback (works always)
        if hma_pullback_long:
            long_score += 2
            entry_reason = "hma_pullback"
        
        # Path 4: RSI oversold + daily bullish (simple mean-reversion)
        if rsi_oversold and daily_bullish:
            long_score += 2
            entry_reason = "rsi_oversold"
        
        # Trend alignment bonus
        if daily_bullish:
            long_score += 1
        if weekly_bullish:
            long_score += 1
        
        # Enter long if score >= 4 (LENIENT threshold)
        if long_score >= 4:
            if weekly_bullish and daily_bullish:
                new_signal = HIGH_CONV_SIZE
            elif daily_bullish:
                new_signal = BASE_SIZE
            else:
                new_signal = LOW_CONV_SIZE
        
        # SHORT ENTRY - Multiple paths
        short_score = 0
        
        # Path 1: Fisher reversal in range regime
        if fisher_short and is_range_regime:
            short_score += 3
            entry_reason = "fisher_range_short"
        
        # Path 2: Donchian breakout in trend regime
        if donchian_breakout_short and is_trend_regime:
            short_score += 3
            entry_reason = "donchian_trend_short"
        
        # Path 3: HMA pullback (works always)
        if hma_pullback_short:
            short_score += 2
            entry_reason = "hma_pullback_short"
        
        # Path 4: RSI overbought + daily bearish
        if rsi_overbought and daily_bearish:
            short_score += 2
            entry_reason = "rsi_overbought"
        
        # Trend alignment bonus
        if daily_bearish:
            short_score += 1
        if weekly_bearish:
            short_score += 1
        
        # Enter short if score >= 4 (LENIENT threshold)
        if short_score >= 4:
            if weekly_bearish and daily_bearish:
                new_signal = -HIGH_CONV_SIZE
            elif daily_bearish:
                new_signal = -BASE_SIZE
            else:
                new_signal = -LOW_CONV_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~360 hours = 15 days on 12h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if daily_bullish and rsi_14[i] < 50:
                new_signal = LOW_CONV_SIZE
            elif daily_bearish and rsi_14[i] > 50:
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
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === REGIME CHANGE EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit long if regime switches from range to strong trend down
            if position_side > 0 and is_trend_regime and daily_bearish:
                regime_exit = True
            # Exit short if regime switches from range to strong trend up
            if position_side < 0 and is_trend_regime and daily_bullish:
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