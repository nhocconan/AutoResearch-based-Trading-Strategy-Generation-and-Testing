#!/usr/bin/env python3
"""
Experiment #966: 1d Primary + 1w HTF — Fisher Transform + CHOP Regime + HMA Trend

Hypothesis: Daily timeframe with Ehlers Fisher Transform for reversal entries,
Choppiness Index regime filter, and Weekly HMA for trend bias will outperform
in mixed 2022-2025 markets (bull 2021, crash 2022, bear/range 2023-2025).

Key innovations:
1. Fisher Transform (period=9): Normalizes price to -1.5 to +1.5 range
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
   - Catches bear market rally reversals better than RSI
2. CHOP(14) regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
3. 1w HMA(21) for weekly trend bias (only trade with weekly direction)
4. 1d HMA(50) for intermediate trend filter
5. ATR(14) 2.5x trailing stop for risk management
6. LOOSE entry thresholds to guarantee 20-50 trades/year on 1d

Why this should work:
- Fisher Transform excels in bear/range markets (2022-2025 test period)
- CHOP filter avoids trend strategies during 2022 bottom whipsaw
- Weekly bias prevents counter-trend trades in strong moves
- 1d timeframe = 20-50 trades/year target (low fee drag)
- Discrete signal sizes minimize churn costs

Entry conditions (LOOSE to guarantee trades):
- LONG = 1w bull + (CHOP<38 + Fisher>-1.5 cross OR CHOP>61 + Fisher<-1.0)
- SHORT = 1w bear + (CHOP<38 + Fisher<+1.5 cross OR CHOP>61 + Fisher>+1.0)
- Relaxed Fisher thresholds for more signals

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag vs EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        w_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / w_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution (-1.5 to +1.5 typical range)
    Excellent for spotting reversals in bear/range markets
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest > lowest:
            normalized = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * (fisher[i-1] if not np.isnan(fisher[i-1]) else 0.0)
            normalized = np.clip(normalized, -0.999, 0.999)
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            trigger[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
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
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        if highest_high > lowest_low and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / tr_sum) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d_50 = calculate_hma(close, period=50)
    hma_1d_21 = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_50[i]) or np.isnan(hma_1d_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND FILTER ===
        trend_1d_bull = hma_1d_21[i] > hma_1d_50[i]
        trend_1d_bear = hma_1d_21[i] < hma_1d_50[i]
        
        # === REGIME DETECTION (CHOP) ===
        is_trending = chop_14[i] < 38.2
        is_ranging = chop_14[i] > 61.8
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = (fisher_trigger[i-1] <= -1.5) and (fisher[i] > -1.5)
        fisher_cross_short = (fisher_trigger[i-1] >= 1.5) and (fisher[i] < 1.5)
        
        # Relaxed Fisher levels for more trades
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === RSI FILTER (avoid extreme counter-trend) ===
        rsi_neutral = (rsi_14[i] > 30) and (rsi_14[i] < 70)
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE + LOOSE) ===
        desired_signal = 0.0
        
        # LONG entries - LOOSE conditions for guaranteed trades
        if htf_1w_bull:
            if is_trending and fisher_cross_long:
                # Trend regime: Fisher reversal entry
                desired_signal = SIZE_STRONG
            elif is_ranging and fisher_oversold:
                # Range regime: Fisher mean reversion
                desired_signal = SIZE_BASE
            elif trend_1d_bull and fisher[i] < -0.5 and rsi_14[i] < 50:
                # Pullback in uptrend
                desired_signal = SIZE_BASE
            elif fisher_cross_long and rsi_14[i] < 60:
                # Any Fisher cross with RSI confirmation
                desired_signal = SIZE_BASE
        
        # SHORT entries - LOOSE conditions for guaranteed trades
        elif htf_1w_bear:
            if is_trending and fisher_cross_short:
                # Trend regime: Fisher reversal entry
                desired_signal = -SIZE_STRONG
            elif is_ranging and fisher_overbought:
                # Range regime: Fisher mean reversion
                desired_signal = -SIZE_BASE
            elif trend_1d_bear and fisher[i] > 0.5 and rsi_14[i] > 50:
                # Pullback in downtrend
                desired_signal = -SIZE_BASE
            elif fisher_cross_short and rsi_14[i] > 40:
                # Any Fisher cross with RSI confirmation
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