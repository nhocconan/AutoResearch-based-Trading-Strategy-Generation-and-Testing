#!/usr/bin/env python3
"""
Experiment #1328: 30m Primary + 4h/1d HTF — Funding Z-Score + Choppiness + Connors RSI

Hypothesis: Lower TF strategies fail due to fee drag from too many trades.
This strategy uses MULTIPLE confluence filters to generate ONLY 30-80 trades/year:
1. 4h HMA for macro trend direction (align with mtf_data)
2. 1d Funding Rate Z-Score for contrarian edge (proven for BTC/ETH)
3. Choppiness Index (14) for regime detection (>55=range, <45=trend)
4. Connors RSI (3-component) for entry timing
5. Session filter (8-20 UTC only) + Volume filter (>0.8x avg)
6. ATR(14) trailing stop at 2.0x for risk management

Key insight from research: Funding rate mean reversion has Sharpe 0.8-1.5 through 2022 crash.
Combining with HTF trend + CRSI entries should work in both bull and bear markets.

Target: 30-80 trades/year, Sharpe > 0.612, trades >= 40 train, >= 5 test
Timeframe: 30m
Size: 0.20-0.25 discrete levels (smaller for lower TF to reduce fee impact)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_funding_chop_crsi_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        choppiness[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - 3-component mean reversion indicator
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Long: CRSI < 10, Short: CRSI > 90
    """
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi_close[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_close[:rsi_period] = np.nan
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask2 = streak_loss_smooth > 1e-10
    rsi_streak[mask2] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask2] / streak_loss_smooth[mask2]))
    rsi_streak[:streak_period+5] = np.nan
    
    # Component 3: PercentRank(100) - where current price ranks in last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < close[i])
            percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine components
    crsi = np.full(n, np.nan)
    valid = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi_close[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i-period+1:i+1])
    return vol_sma

def calculate_funding_zscore(funding_data, prices, period=30):
    """Z-Score of funding rate for contrarian signal
    Load funding data from parquet, calculate rolling z-score
    """
    try:
        # Try to load funding data - if not available, return neutral
        funding_path = f"data/processed/funding/{prices['symbol'].iloc[0] if 'symbol' in prices.columns else 'BTCUSDT'}.parquet"
        df_funding = pd.read_parquet(funding_path)
        
        if len(df_funding) < period + 10:
            return np.zeros(len(prices))
        
        # Align funding to prices timeframe
        funding_rates = df_funding['funding_rate'].values
        
        # Calculate z-score
        n_prices = len(prices)
        zscore = np.zeros(n_prices)
        
        for i in range(period, min(len(funding_rates), n_prices)):
            window = funding_rates[i-period:i]
            if len(window) >= period:
                mean_funding = np.mean(window)
                std_funding = np.std(window)
                if std_funding > 1e-10:
                    zscore[i] = (funding_rates[i] - mean_funding) / std_funding
        
        # Align to prices length
        if len(zscore) < n_prices:
            zscore = np.pad(zscore, (0, n_prices - len(zscore)), mode='constant')
        elif len(zscore) > n_prices:
            zscore = zscore[:n_prices]
        
        return zscore
    
    except Exception:
        # If funding data not available, return neutral z-score
        return np.zeros(len(prices))

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Try to get funding z-score (may not be available for all symbols)
    try:
        funding_zscore = calculate_funding_zscore(None, prices, period=30)
    except:
        funding_zscore = np.zeros(n)
    
    # Calculate primary (30m) indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22  # Smaller size for lower TF to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (> 0.8x average) ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === REGIME DETECTION (Choppiness) ===
        chop = choppiness[i]
        is_range = chop > 55.0  # Range/choppy market
        is_trend = chop < 45.0  # Trending market
        
        # === HTF TREND BIAS ===
        trend_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        
        # === FUNDING Z-SCORE CONTRARIAN ===
        fund_z = funding_zscore[i]
        funding_extreme_long = fund_z < -1.5  # Very negative funding = long opportunity
        funding_extreme_short = fund_z > 1.5  # Very positive funding = short opportunity
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_neutral = 35.0 <= crsi[i] <= 65.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Multiple confluence required (3+ filters)
        long_confluence = 0
        
        if in_session:
            long_confluence += 1
        if volume_ok:
            long_confluence += 1
        if trend_bull or is_range:
            long_confluence += 1
        if crsi_oversold or funding_extreme_long:
            long_confluence += 1
        
        # Need at least 3 confluence for long
        if long_confluence >= 3:
            if crsi_oversold and (trend_bull or is_range):
                desired_signal = BASE_SIZE
            elif funding_extreme_long and crsi[i] < 40.0:
                desired_signal = BASE_SIZE
            elif is_range and crsi_oversold and volume_ok:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Multiple confluence required (3+ filters)
        short_confluence = 0
        
        if in_session:
            short_confluence += 1
        if volume_ok:
            short_confluence += 1
        if trend_bear or is_range:
            short_confluence += 1
        if crsi_overbought or funding_extreme_short:
            short_confluence += 1
        
        # Need at least 3 confluence for short
        if short_confluence >= 3:
            if crsi_overbought and (trend_bear or is_range):
                desired_signal = -BASE_SIZE
            elif funding_extreme_short and crsi[i] > 60.0:
                desired_signal = -BASE_SIZE
            elif is_range and crsi_overbought and volume_ok:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals