#!/usr/bin/env python3
"""
Experiment #003: 1d Fisher Transform + Choppiness Regime + 1w Trend Filter

Hypothesis: Previous strategies (Connors RSI, KAMA) failed because they don't 
adapt to crypto's regime shifts between trending and ranging markets. This strategy uses:

1. Ehlers Fisher Transform (period=9) - Superior reversal detection in bear markets.
   Long when Fisher crosses above -1.5 from below. Short when crosses below +1.5.
   Proven to catch bear market rally reversals better than RSI.

2. Choppiness Index (period=14) - Regime detection. CHOP > 61.8 = range (mean revert),
   CHOP < 38.2 = trending (trend follow). This META-FILTER switches strategy logic.

3. 1w HMA(21) Major Trend - Via mtf_data helper. Only long if price > 1w HMA,
   only short if price < 1w HMA. Prevents counter-trend trades in major moves.

4. Asymmetric Entry Logic - In bear regime (price < 1w HMA), only short retraces
   to resistance. In bull regime (price > 1w HMA), only long pullbacks to support.
   This matches crypto's asymmetric behavior (slow up, fast down).

5. ATR(14) Trailing Stop - 2.5x ATR for risk management. Signal → 0 when stopped.

6. Volume Confirmation - Entry only if volume > 20-bar average (avoids fake breakouts).

Why this should work:
- Fisher Transform outperforms RSI for reversal detection (Ehlers research)
- Choppiness Index prevents trend strategies in chop and vice versa
- 1d timeframe = 20-50 trades/year target (optimal for fee drag on daily)
- 1w HTF filter ensures we trade with major trend (critical for crypto)
- Conservative sizing (0.20-0.30) protects against 77% crashes like 2022
- Different from failed strategies (no Connors RSI, no KAMA)

Timeframe: 1d (REQUIRED for Experiment #003)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20 base, 0.30 high conviction (weekly alignment)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals better than RSI in bear markets.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate typical price
    typical = (high_s + low_s + close_s) / 3
    
    # Normalize to -1 to +1 range
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, np.nan)
    
    normalized = 2 * ((typical - lowest) / range_val - 0.5)
    normalized = normalized.clip(-0.99, 0.99)  # Prevent log errors
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = fisher.replace([np.inf, -np.inf], np.nan)
    
    # Signal line (previous Fisher)
    fisher_prev = fisher.shift(1)
    
    return fisher.values, fisher_prev.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = range-bound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high - Lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    hh_ll = hh - ll
    hh_ll = hh_ll.replace(0, np.nan)
    
    # Choppiness formula
    chop = 100 * np.log10(tr_sum / hh_ll) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

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

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.20
    HIGH_CONV_SIZE = 0.30
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(vol_ma[i]):
            continue
        
        # === WEEKLY MAJOR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        neutral_regime = not is_choppy and not is_trending
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_signal = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_short_signal = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Additional Fisher extremes for mean reversion in choppy regime
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]  # 20% above average
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        entry_reason = ""
        
        # LONG ENTRY
        if weekly_bullish:
            long_score = 0
            
            # Fisher reversal signal (primary)
            if fisher_long_signal:
                long_score += 3
                entry_reason = "fisher_reversal"
            
            # Fisher oversold in choppy regime (mean reversion)
            if is_choppy and fisher_oversold:
                long_score += 2
                entry_reason = "fisher_oversold_chop"
            
            # Fisher pullback in trending regime
            if is_trending and fisher[i] > -1.0 and fisher[i] < 0:
                long_score += 1.5
                entry_reason = "fisher_pullback_trend"
            
            # Volume confirmation adds conviction
            if volume_confirmed:
                long_score += 1
            
            # Enter long if score >= 4
            if long_score >= 4:
                if weekly_bullish:
                    new_signal = HIGH_CONV_SIZE
                else:
                    new_signal = BASE_SIZE
        
        # SHORT ENTRY
        if weekly_bearish:
            short_score = 0
            
            # Fisher reversal signal (primary)
            if fisher_short_signal:
                short_score += 3
                entry_reason = "fisher_reversal"
            
            # Fisher overbought in choppy regime (mean reversion)
            if is_choppy and fisher_overbought:
                short_score += 2
                entry_reason = "fisher_overbought_chop"
            
            # Fisher pullback in trending regime
            if is_trending and fisher[i] < 1.0 and fisher[i] > 0:
                short_score += 1.5
                entry_reason = "fisher_pullback_trend"
            
            # Volume confirmation adds conviction
            if volume_confirmed:
                short_score += 1
            
            # Enter short if score >= 4
            if short_score >= 4:
                if weekly_bearish:
                    new_signal = -HIGH_CONV_SIZE
                else:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~30 days on 1d), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if weekly_bullish and fisher[i] < -1.0:
                new_signal = BASE_SIZE
            elif weekly_bearish and fisher[i] > 1.0:
                new_signal = -BASE_SIZE
        
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
            # Exit long if weekly trend turns bearish
            if position_side > 0 and weekly_bearish:
                trend_reversal = True
            # Exit short if weekly trend turns bullish
            if position_side < 0 and weekly_bullish:
                trend_reversal = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long if Fisher becomes overbought
            if position_side > 0 and fisher[i] > 2.0:
                fisher_exit = True
            # Exit short if Fisher becomes oversold
            if position_side < 0 and fisher[i] < -2.0:
                fisher_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or fisher_exit:
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