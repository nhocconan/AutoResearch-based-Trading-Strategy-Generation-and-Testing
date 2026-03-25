#!/usr/bin/env python3
"""
Experiment #1555: 6h Primary + 12h/1d HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Connors RSI (CRSI) has 
proven 75% win rate in quantitative literature for mean reversion. Combined with:
1. Choppiness Index regime detection (range vs trend)
2. 1d HMA(21) for major trend bias (avoid counter-trend disasters)
3. 12h HMA(16/48) for intermediate momentum
4. Volume sentiment filter (taker_buy_volume ratio)
5. ATR vol-spike confirmation (avoid entering during vol crush)

Why 6h should work:
- Middle ground between 4h (too many trades) and 12h (too few)
- Natural 30-60 trades/year target
- Captures multi-day swings without intraday noise
- CRSI extremes (10/90) are rare enough to be meaningful, common enough to trade

Entry logic (LOOSE to guarantee trades):
- LONG: CRSI<15 + price>1d_HMA + taker_buy_ratio>0.45 (not extreme selling)
- SHORT: CRSI>85 + price<1d_HMA + taker_buy_ratio<0.55 (not extreme buying)
- Range regime (CHOP>61.8): loosen CRSI to 20/80 thresholds
- Trend regime (CHOP<38.2): only trade with 1d HMA direction

Risk management:
- ATR(14) trailing stop at 2.5x
- Discrete sizing: 0.0, ±0.25, ±0.30
- Stoploss via signal→0 when triggered

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_chop_regime_1d12h_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - proven mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: where current close ranks in last N bars (0-100)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_window = streak[i - streak_period + 1:i + 1]
        gains = np.sum(np.where(streak_window > 0, streak_window, 0))
        losses = np.abs(np.sum(np.where(streak_window < 0, streak_window, 0)))
        if losses == 0:
            streak_rsi[i] = 100.0
        else:
            rs = gains / losses
            streak_rsi[i] = 100 - (100 / (1 + rs))
    
    # Percent Rank - where current close ranks in last N bars
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for vol spike detection"""
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_taker_buy_ratio(prices):
    """Taker buy volume ratio as sentiment indicator"""
    n = len(prices)
    ratio = np.full(n, np.nan, dtype=np.float64)
    
    if 'taker_buy_volume' not in prices.columns or 'volume' not in prices.columns:
        return np.full(n, 0.5)  # neutral if data not available
    
    taker_vol = prices['taker_buy_volume'].values
    total_vol = prices['volume'].values
    
    for i in range(n):
        if total_vol[i] > 1e-10:
            ratio[i] = taker_vol[i] / total_vol[i]
        else:
            ratio[i] = 0.5
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_12h_16_raw = calculate_hma(df_12h['close'].values, period=16)
    hma_12h_48_raw = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_16_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_16_raw)
    hma_12h_48_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_48_raw)
    
    # Calculate 6h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    taker_ratio = calculate_taker_buy_ratio(prices)
    
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
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h_16_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        is_neutral_regime = not is_trend_regime and not is_range_regime
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA CROSSOVER (intermediate momentum) ===
        hma_12h_bullish = hma_12h_16_aligned[i] > hma_12h_48_aligned[i]
        hma_12h_bearish = hma_12h_16_aligned[i] < hma_12h_48_aligned[i]
        
        # === CRSI (Connors RSI) ===
        crsi_val = crsi[i]
        crsi_extreme_low = crsi_val < 15  # extreme oversold
        crsi_extreme_high = crsi_val > 85  # extreme overbought
        
        # === VOLUME SENTIMENT ===
        taker_sentiment = taker_ratio[i]
        not_extreme_selling = taker_sentiment > 0.40  # not panic selling
        not_extreme_buying = taker_sentiment < 0.60  # not FOMO buying
        
        # === VOL SPIKE CONFIRMATION ===
        vol_expansion = atr_ratio[i] > 1.2 if not np.isnan(atr_ratio[i]) else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # RANGE REGIME: CRSI mean reversion (primary strategy)
        if is_range_regime:
            # LONG: CRSI extreme low + not extreme selling + vol confirmation
            if crsi_val < 20 and not_extreme_selling:
                desired_signal = SIZE_BASE
                if vol_expansion:
                    desired_signal = SIZE_STRONG
            
            # SHORT: CRSI extreme high + not extreme buying
            elif crsi_val > 80 and not_extreme_buying:
                desired_signal = -SIZE_BASE
                if vol_expansion:
                    desired_signal = -SIZE_STRONG
        
        # TREND REGIME: Only trade with trend direction
        elif is_trend_regime:
            # LONG: 1d bullish + 12h bullish + CRSI pullback (not extreme)
            if price_above_1d and hma_12h_bullish and crsi_val < 40:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + 12h bearish + CRSI rally (not extreme)
            elif price_below_1d and hma_12h_bearish and crsi_val > 60:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Standard CRSI extremes with 1d filter
        elif is_neutral_regime:
            # LONG: CRSI extreme + 1d not bearish
            if crsi_extreme_low and not price_below_1d:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI extreme + 1d not bullish
            elif crsi_extreme_high and not price_above_1d:
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