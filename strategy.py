#!/usr/bin/env python3
"""
Experiment #1264: 12h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: 12h timeframe with regime-switching logic will outperform pure trend strategies.
Key insight from research: Choppiness Index is the BEST meta-filter for bear/range markets.
This strategy switches between mean-reversion (chop) and trend-following (trend) based on CHOP.

Why this should work:
1. CHOP(14) > 61.8 = range market → Connors RSI mean reversion (75% win rate documented)
2. CHOP(14) < 38.2 = trend market → HMA trend following with momentum
3. 1d HMA(21) for major regime bias (only trade with daily trend)
4. 1w HMA(21) for ultra-long term regime filter
5. ATR(14) 2.5x trailing stop for risk management
6. LOOSE entry conditions to guarantee 30-50 trades/year on 12h

Entry logic (LOOSE to guarantee trades):
- RANGE (CHOP>61.8): Long when CRSI<15 + price>SMA200, Short when CRSI>85 + price<SMA200
- TREND (CHOP<38.2): Long when 12h_HMA rising + 1d_HMA bullish, Short when opposite
- NEUTRAL (38.2<=CHOP<=61.8): Only trade strong momentum (ROC>10 or <-10)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_hma_1d1w_v1"
timeframe = "12h"
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
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    
    gain[1:] = np.maximum(delta, 0)
    loss[1:] = np.maximum(-delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        if len(streak_window) > 0:
            pos_count = np.sum(streak_window > 0)
            streak_rsi[i] = (pos_count / len(streak_window)) * 100.0
    
    # Percent Rank (100) - where current return ranks in last 100 bars
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and not np.all(np.isnan(returns)):
            current_return = returns[-1]
            valid_returns = returns[~np.isnan(returns)]
            if len(valid_returns) > 0:
                pct_rank[i] = (np.sum(valid_returns < current_return) / len(valid_returns)) * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + pct_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        window_high = high[i-period+1:i+1]
        window_low = low[i-period+1:i+1]
        window_close = close[i-period+1:i+1]
        
        if np.any(np.isnan(window_high)) or np.any(np.isnan(window_low)):
            continue
        
        highest_high = np.nanmax(window_high)
        lowest_low = np.nanmin(window_low)
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_12h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    
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
    min_bars = 250  # Need enough for CRSI rank_period + SMA200
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop[i] > 61.8
        trending_regime = chop[i] < 38.2
        # neutral_regime = not choppy and not trending
        
        # === TREND DIRECTION ===
        # 12h HMA slope (compare to 5 bars ago for stability)
        hma_12h_slope = 0.0
        if i >= 5 and not np.isnan(hma_12h[i-5]):
            hma_12h_slope = hma_12h[i] - hma_12h[i-5]
        
        # 1d HMA bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA bias (ultra long term)
        price_above_1w = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        price_below_1w = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # Price vs SMA200
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        if choppy_regime:
            # MEAN REVERSION: Connors RSI extremes
            # LONG: CRSI < 15 (oversold) + price above SMA200
            if crsi[i] < 20 and price_above_sma200:
                if crsi[i] < 12:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: CRSI > 85 (overbought) + price below SMA200
            elif crsi[i] > 80 and price_below_sma200:
                if crsi[i] > 88:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        elif trending_regime:
            # TREND FOLLOWING: HMA direction + HTF confirmation
            # LONG: 12h HMA rising + 1d bullish + 1w neutral/bullish
            if hma_12h_slope > 0 and price_above_1d:
                if price_above_1w:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: 12h HMA falling + 1d bearish + 1w neutral/bearish
            elif hma_12h_slope < 0 and price_below_1d:
                if price_below_1w:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        else:
            # NEUTRAL regime - only trade strong momentum
            roc = roc_10[i]
            if not np.isnan(roc):
                if roc > 8.0 and price_above_1d:
                    desired_signal = SIZE_BASE
                elif roc < -8.0 and price_below_1d:
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