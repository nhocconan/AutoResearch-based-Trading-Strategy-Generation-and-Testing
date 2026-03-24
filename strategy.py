#!/usr/bin/env python3
"""
Experiment #979: 1h Primary + 4h/12h HTF — Fisher Transform + CHOP Regime + Session Filter

Hypothesis: 1h timeframe with Ehlers Fisher Transform entries + Choppiness regime filter
+ 4h HMA trend bias will outperform in mixed 2022-2025 markets with controlled trade frequency.

Key innovations:
1. Ehlers Fisher Transform (period=9): Transforms prices to Gaussian distribution for better
   reversal detection. Long when Fisher crosses above -1.5, short when crosses below +1.5.
2. Choppiness Index (14): CHOP > 61.8 = range (use Fisher reversals), CHOP < 38.2 = trend
   (use HMA pullback entries). Best meta-filter for bear/range markets.
3. 4h HMA(21) for intermediate trend bias (smoother than EMA, less lag than SMA)
4. 12h ADX(14) for trend strength confirmation (ADX > 25 = trend, ADX < 20 = range)
5. Session filter: 08-20 UTC only (avoid low liquidity Asian session false signals)
6. ATR(14) 2.5x trailing stop for risk management
7. Conservative sizing: 0.20-0.30 discrete to survive 2022-style crashes

Why this should work:
- Fisher Transform catches reversals in bear rallies better than RSI (proven in literature)
- CHOP + ADX dual regime filter avoids trend strategies in choppy 2022 bottom
- 4h HMA provides smooth trend bias without 12h lag
- Session filter reduces false signals by ~40% during low volume periods
- 1h captures intraday swings with HTF direction = 40-80 trades/year target

Entry conditions (balanced for trade frequency):
- LONG = 4h bull + (CHOP>61 + Fisher<-1.5 cross OR CHOP<38 + price>HMA + ADX>25)
- SHORT = 4h bear + (CHOP>61 + Fisher>+1.5 cross OR CHOP<38 + price<HMA + ADX>25)
- Session: 08-20 UTC only

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Transforms prices to near-Gaussian distribution for better reversal detection.
    
    Formula:
    1. Price = (0.33 * 2 * ((H-L)/(H+L)) + 0.67 * PrevPrice)
    2. Fisher = 0.5 * ln((1+Price)/(1-Price))
    3. Trigger = PrevFisher
    
    Long: Fisher crosses above -1.5
    Short: Fisher crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate normalized price
    price = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hl_range = high[i] + low[i]
        if hl_range > 1e-10:
            price[i] = 0.33 * 2.0 * ((high[i] - low[i]) / hl_range) + 0.67 * price[i-1]
        else:
            price[i] = price[i-1]
    
    # Clamp price to avoid log(0) or log(negative)
    price = np.clip(price, -0.999, 0.999)
    
    # Calculate Fisher
    for i in range(period, n):
        fisher[i] = 0.5 * np.log((1.0 + price[i]) / (1.0 - price[i]) + 1e-10)
        if i > 0:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures whether market is trending or choppy/ranging.
    CHOP > 61.8 = range/choppy (use mean reversion)
    CHOP < 38.2 = trending (use trend following)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if highest_high > lowest_low and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / tr_sum) / np.log10(period)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction).
    ADX > 25 = strong trend
    ADX < 20 = weak trend / ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method (EMA with alpha = 1/period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_hma(close, period):
    """Hull Moving Average - smoother than EMA, less lag than SMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # Convert milliseconds to seconds, then to datetime
    hours = np.zeros(len(open_time_array), dtype=np.int32)
    for i in range(len(open_time_array)):
        ts_sec = open_time_array[i] / 1000.0
        hours[i] = int((ts_sec % 86400) / 3600)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract UTC hour for session filter
    utc_hours = get_hour_from_open_time(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate 1h indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === HTF BIAS (4h HMA + 12h ADX) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_trend_strong = adx_12h_aligned[i] > 25.0
        htf_trend_weak = adx_12h_aligned[i] < 20.0
        
        # === REGIME DETECTION (CHOP) ===
        is_ranging = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = False
        fisher_short_cross = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_trigger[i-1]):
            # Long: Fisher crosses above -1.5 from below
            fisher_long_cross = (fisher_trigger[i-1] <= -1.5) and (fisher[i] > -1.5)
            # Short: Fisher crosses below +1.5 from above
            fisher_short_cross = (fisher_trigger[i-1] >= 1.5) and (fisher[i] < 1.5)
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # LONG entries (require 4h bull bias)
        if htf_4h_bull and in_session:
            if is_ranging and fisher_long_cross:
                # Range regime: Fisher reversal entry
                desired_signal = SIZE_BASE
            elif is_trending and htf_trend_strong:
                # Trend regime: pullback to HMA with strong ADX
                if close[i] > hma_4h_aligned[i] * 0.995:  # Near or above HMA
                    desired_signal = SIZE_STRONG
            elif fisher[i] < -1.0 and htf_trend_weak:
                # Weak trend + oversold Fisher
                desired_signal = SIZE_BASE
        
        # SHORT entries (require 4h bear bias)
        elif htf_4h_bear and in_session:
            if is_ranging and fisher_short_cross:
                # Range regime: Fisher reversal entry
                desired_signal = -SIZE_BASE
            elif is_trending and htf_trend_strong:
                # Trend regime: pullback to HMA with strong ADX
                if close[i] < hma_4h_aligned[i] * 1.005:  # Near or below HMA
                    desired_signal = -SIZE_STRONG
            elif fisher[i] > 1.0 and htf_trend_weak:
                # Weak trend + overbought Fisher
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals