#!/usr/bin/env python3
"""
Experiment #437: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime Switch

Hypothesis: Daily timeframe needs fewer but higher-quality signals. Previous 1d strategies
(#427, #433) failed due to overly strict filters or wrong regime logic. This strategy:

1. Uses Fisher Transform for reversal detection (proven on daily charts, catches bear rallies)
2. Choppiness Index regime switch: mean-revert in chop, trend-follow otherwise
3. 1w HMA for major trend bias (soft filter, not hard block)
4. VERY relaxed CRSI thresholds for daily (<25/>75 vs <20/>80) to ensure trade frequency
5. Volume confirmation on breakouts (avoid fakeouts)
6. Position size 0.30 with ATR-based stoploss

Target: 80-200 trades over 4-year train (20-50/year), Sharpe > 0.612, DD < -40%
Key insight: 1d data has fewer signals, so each signal must be high-confidence but not rare.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_crsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (close - lowest) / (highest - lowest) - 0.67
    Catches reversals in bear/range markets.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1]) if 'high' in dir() else np.nanmax(close[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1]) if 'low' in dir() else np.nanmin(close[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        x = 0.67 * (close[i] - lowest) / price_range - 0.67
        x = np.clip(x, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_fisher_from_arrays(close, high, low, period=9):
    """Fisher Transform with explicit high/low arrays."""
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        x = 0.67 * (close[i] - lowest) / price_range - 0.67
        x = np.clip(x, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    
    # RSI(3) component
    rsi = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank component
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    # Combine
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_volume_ma(volume, period=20):
    """Calculate Volume Moving Average."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    fisher, fisher_prev = calculate_fisher_from_arrays(close, high, low, period=9)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 1d
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 55.0  # Range market
        regime_trend = chop[i] < 45.0  # Trending market
        
        # === TREND BIAS (1w HMA) — SOFT FILTER ONLY ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS (Reversals) ===
        fisher_oversold = fisher[i] < -1.5  # Strong reversal signal
        fisher_overbought = fisher[i] > 1.5  # Strong reversal signal
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5 if not np.isnan(fisher_prev[i]) else False
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5 if not np.isnan(fisher_prev[i]) else False
        
        # === CRSI SIGNALS (Mean Reversion) — RELAXED FOR DAILY ===
        crsi_oversold = crsi[i] < 25.0  # Relaxed for 1d (vs <20)
        crsi_overbought = crsi[i] > 75.0  # Relaxed for 1d (vs >80)
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / (vol_ma[i] + 1e-10) if not np.isnan(vol_ma[i]) else 1.0
        volume_confirmed = vol_ratio > 1.2  # 20% above average
        
        # === VOL FILTER ===
        vol_ratio_atr = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio_atr > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio_atr > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 55) — MEAN REVERSION ===
        if regime_chop:
            # Long: Fisher oversold cross OR CRSI oversold
            if fisher_cross_up or crsi_oversold:
                if price_above_sma200 or crsi_extreme_oversold:  # SMA200 filter relaxed with extreme CRSI
                    desired_signal = position_size
            elif crsi_extreme_oversold:
                desired_signal = position_size
            
            # Short: Fisher overbought cross OR CRSI overbought
            if fisher_cross_down or crsi_overbought:
                if desired_signal == 0 and (price_below_sma200 or crsi_extreme_overbought):
                    desired_signal = -position_size
            elif crsi_extreme_overbought:
                desired_signal = -position_size
        
        # === REGIME 2: TRENDING (CHOP < 45) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout with volume OR HMA bullish + Fisher support
            if donchian_breakout_long and volume_confirmed:
                desired_signal = position_size
            elif hma_bullish and fisher[i] > -1.0:
                desired_signal = position_size * 0.8
            
            # Short: Donchian breakdown with volume OR HMA bearish + Fisher support
            if donchian_breakout_short and volume_confirmed and desired_signal == 0:
                desired_signal = -position_size
            elif hma_bearish and fisher[i] < 1.0:
                desired_signal = -position_size * 0.8
        
        # === REGIME 3: TRANSITION (45-55) — REDUCED SIZE, ONLY EXTREMES ===
        else:
            # Only extreme Fisher or CRSI signals
            if fisher_cross_up or crsi_extreme_oversold:
                desired_signal = position_size * 0.5
            elif fisher_cross_down or crsi_extreme_overbought:
                desired_signal = -position_size * 0.5
        
        # === HTF BIAS MODIFIER (1w HMA) — SOFT, NOT HARD ===
        # Reduce opposite-direction trades but don't block completely
        if desired_signal > 0 and price_below_hma_1w:
            desired_signal = desired_signal * 0.5  # Reduce long when 1w bearish
        if desired_signal < 0 and price_above_hma_1w:
            desired_signal = desired_signal * 0.5  # Reduce short when 1w bullish
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or fisher[i] > -1.0):
                desired_signal = position_size
            elif position_side < 0 and (hma_bearish or fisher[i] < 1.0):
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals