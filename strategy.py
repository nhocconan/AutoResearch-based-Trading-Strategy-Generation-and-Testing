#!/usr/bin/env python3
"""
Experiment #1192: 12h Primary + 1d/1w HTF — Adaptive Regime with Funding Rate Filter

Hypothesis: Recent 12h strategies failed due to overly strict entry conditions causing 0 trades.
This version focuses on generating sufficient trade frequency while maintaining quality:
- Primary regime: Choppiness Index (61.8 chop / 38.2 trend)
- Mean reversion: CRSI < 25 / > 75 (relaxed from 20/80 for more trades)
- Trend: Donchian(20) breakout + 1d HMA alignment
- NEW: Funding rate z-score filter for BTC/ETH contrarian edge
- Relaxed BB confirmation (optional, not required)
- Ensure ALL symbols can generate trades (no SOL-only bias)

Key improvements from #1186 (Sharpe=-0.057):
1. CRSI thresholds: 20/80 → 25/75 (more entry opportunities)
2. CHOP transition zone handling: enter on extremes even in transition
3. Add funding rate z-score for BTC/ETH mean reversion edge
4. Simplified trend entry (breakout OR HMA cross, not both)
5. Lower minimum lookback (100 vs 150) for earlier signals

Target: 25-45 trades/year per symbol, Sharpe > 0.612
Position Size: 0.28 discrete (balanced risk/reward)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_adaptive_regime_crsi_funding_donchian_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[i-streak_period+1:i+1] > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100.0
    
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window <= current) / rank_period
        pct_rank[i] = rank * 100.0
    
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout indicator."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_bb(close, period=20, std_mult=2.0):
    """Bollinger Bands — mean reversion levels."""
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mid[i] = np.mean(window)
        std = np.std(window)
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
    
    return mid, upper, lower

def calculate_funding_zscore(prices, symbol, lookback=90):
    """
    Funding Rate Z-Score — contrarian indicator for BTC/ETH.
    High positive funding = crowded longs = short signal
    High negative funding = crowded shorts = long signal
    """
    n = len(prices)
    zscore = np.full(n, np.nan)
    
    try:
        import os
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
            if len(funding_df) > 0 and 'funding_rate' in funding_df.columns:
                funding_rates = funding_df['funding_rate'].values
                min_len = min(n, len(funding_rates))
                funding_aligned = np.full(n, np.nan)
                funding_aligned[-min_len:] = funding_rates[-min_len:]
                
                for i in range(lookback, n):
                    window = funding_aligned[i-lookback:i]
                    if not np.any(np.isnan(window)):
                        mean_f = np.mean(window)
                        std_f = np.std(window)
                        if std_f > 1e-10:
                            zscore[i] = (funding_aligned[i] - mean_f) / std_f
    except Exception:
        pass
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract symbol from prices for funding data
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if 'symbol' in prices.columns else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for very long-term trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_mid, bb_upper, bb_lower = calculate_bb(close, period=20, std_mult=2.0)
    
    # Calculate funding z-score for contrarian filter
    funding_z = calculate_funding_zscore(prices, symbol, lookback=90)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(bb_lower[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA + 1w HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        very_long_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        very_long_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        is_transition = not is_choppy and not is_trending
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === CONNORS RSI EXTREMES (relaxed for more trades) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === BOLLINGER EXTREMES ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === FUNDING RATE CONTRARIAN (BTC/ETH only) ===
        funding_extreme_long = False
        funding_extreme_short = False
        if not np.isnan(funding_z[i]):
            if symbol.startswith('BTC') or symbol.startswith('ETH'):
                funding_extreme_long = funding_z[i] < -1.5  # crowded shorts
                funding_extreme_short = funding_z[i] > 1.5  # crowded longs
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === TRENDING REGIME: Donchian Breakout + Macro Alignment ===
        if is_trending:
            if macro_bull and breakout_long:
                desired_signal = BASE_SIZE
            elif macro_bear and breakout_short:
                desired_signal = -BASE_SIZE
        
        # === CHOPPY REGIME: Mean Reversion (CRSI + BB + Funding) ===
        elif is_choppy:
            # Long: oversold signals + not strongly bearish macro
            long_conditions = (crsi_oversold or bb_oversold or funding_extreme_long)
            if long_conditions and not macro_bear:
                desired_signal = BASE_SIZE
            # Short: overbought signals + not strongly bullish macro
            short_conditions = (crsi_overbought or bb_overbought or funding_extreme_short)
            if short_conditions and not macro_bull:
                desired_signal = -BASE_SIZE
        
        # === TRANSITION ZONE: Use CRSI extremes + funding ===
        else:
            # More lenient in transition to ensure trade frequency
            if (crsi_oversold or funding_extreme_long) and (macro_bull or very_long_bull):
                desired_signal = BASE_SIZE
            elif (crsi_overbought or funding_extreme_short) and (macro_bear or very_long_bear):
                desired_signal = -BASE_SIZE
        
        # === FALLBACK: Pure CRSI extremes for trade frequency ===
        # If no signal yet but CRSI at extreme, take the trade
        if desired_signal == 0.0:
            if crsi_oversold and not macro_bear:
                desired_signal = BASE_SIZE * 0.5  # Half size for weaker signal
            elif crsi_overbought and not macro_bull:
                desired_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.5:
            desired_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals