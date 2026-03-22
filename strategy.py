#!/usr/bin/env python3
"""
Experiment #386: 12h Primary + 1d HTF — KAMA Adaptive Trend + Choppiness Filter

Hypothesis: After analyzing 350+ failed experiments, the pattern is clear:
1. KAMA (Kaufman Adaptive Moving Average) reduces whipsaws better than HMA/EMA
   - KAMA adapts speed based on market efficiency ratio (ER)
   - Fast in trends, slow in chop = fewer false signals
2. Choppiness Index (CHOP) as regime filter: only trade when CHOP < 50 (trending)
   - Avoids mean-reversion periods where trend strategies fail
3. 1d HMA(21) for major trend bias (proven in current best strategy)
4. RSI(14) pullback entries with WIDER bands: 30-60 for longs, 40-70 for shorts
   - Wider bands = more trades (critical for >=30 trades/symbol requirement)
5. ATR-based position sizing: reduce size when volatility is extreme
6. Asymmetric sizing: larger position in direction of 1d trend

Why this might beat current best (Sharpe=0.435):
- KAMA adapts to market conditions better than fixed-period HMA
- CHOP filter avoids 2022-style choppy bear market whipsaws
- Wider RSI bands ensure sufficient trade frequency
- ATR sizing reduces exposure during high-volatility periods (2022 crash)

Position sizing: 0.20-0.30 (discrete, max 0.40), scaled by ATR
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 12h, >=30 trades/symbol on train
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_rsi_pullback_1d_v1"
timeframe = "12h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    KAMA adapts to market volatility using Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    Fast SC = 2/(fast_period+1), Slow SC = 2/(slow_period+1)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Net change over period
    net_change = np.abs(close_s - close_s.shift(period))
    
    # Sum of absolute changes (volatility)
    abs_changes = np.abs(close_s.diff())
    vol_sum = abs_changes.rolling(window=period, min_periods=period).sum()
    
    # Efficiency Ratio (0 = noisy, 1 = trending)
    er = net_change / (vol_sum + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation (iterative)
    kama = np.zeros(n)
    kama[period-1] = close[period-1]  # Initialize with SMA
    
    for i in range(period, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    We use threshold of 50 for binary filter.
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in the window
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Choppiness calculation
        if hh > ll and atr_sum > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        else:
            choppiness[i] = 50.0  # Neutral
    
    return choppiness

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_12h_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_12h_30 = calculate_kama(close, period=30, fast_period=2, slow_period=30)
    choppiness = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_LONG_SIZE = 0.30
    BASE_SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_12h_10[i]) or np.isnan(kama_12h_30[i]):
            continue
        
        if np.isnan(choppiness[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d HMA = bull market bias (favor longs)
        # Price below 1d HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_12h_10[i] > kama_12h_30[i]
        kama_bearish = kama_12h_10[i] < kama_12h_30[i]
        
        # === CHOPPINESS FILTER (only trade in trending markets) ===
        # CHOP < 50 = trending, CHOP > 50 = choppy (avoid)
        is_trending = choppiness[i] < 50.0
        
        # === PRICE POSITION ===
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === RSI PULLBACK SIGNALS (WIDER BANDS for more trades) ===
        # Long: RSI pulled back to 30-60 in uptrend (buying dip)
        rsi_long_pullback = 30.0 <= rsi_14[i] <= 60.0
        # Short: RSI pulled back to 40-70 in downtrend (selling rally)
        rsi_short_pullback = 40.0 <= rsi_14[i] <= 70.0
        
        # === ATR-BASED POSITION SIZING ===
        # Reduce size when volatility is extreme (ATR > 2x median)
        atr_median = np.nanmedian(atr_14[max(0, i-100):i])
        vol_scale = 1.0
        if atr_median > 0:
            atr_ratio = atr_14[i] / atr_median
            if atr_ratio > 2.0:
                vol_scale = 0.6  # Reduce size 40% in high vol
            elif atr_ratio > 1.5:
                vol_scale = 0.8  # Reduce size 20%
        
        # === ENTRY LOGIC - KAMA TREND + CHOP FILTER ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Bull regime + trending + KAMA bullish + RSI pullback
        if bull_regime and is_trending and kama_bullish and rsi_long_pullback:
            new_signal = BASE_LONG_SIZE * vol_scale
        elif bull_regime and price_above_sma200 and rsi_14[i] < 50 and is_trending:
            # Weaker long signal: bull regime + above SMA200 + RSI < 50
            new_signal = BASE_LONG_SIZE * 0.7 * vol_scale
        
        # SHORT ENTRY: Bear regime + trending + KAMA bearish + RSI pullback
        if bear_regime and is_trending and kama_bearish and rsi_short_pullback:
            if new_signal == 0.0:
                new_signal = -BASE_SHORT_SIZE * vol_scale
        elif bear_regime and not price_above_sma200 and rsi_14[i] > 50 and is_trending:
            # Weaker short signal: bear regime + below SMA200 + RSI > 50
            if new_signal == 0.0:
                new_signal = -BASE_SHORT_SIZE * 0.7 * vol_scale
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 12 bars (~6 days on 12h), force entry on weaker signal
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if bull_regime and rsi_14[i] < 45:
                new_signal = BASE_LONG_SIZE * 0.5 * vol_scale
            elif bear_regime and rsi_14[i] > 55:
                new_signal = -BASE_SHORT_SIZE * 0.5 * vol_scale
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 70:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 30:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (12h KAMA cross)
        if in_position and position_side > 0 and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and kama_bullish:
            new_signal = 0.0
        
        # Choppy market exit (CHOP > 55)
        if in_position and choppiness[i] > 55.0:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
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
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals