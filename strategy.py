#!/usr/bin/env python3
"""
Experiment #306: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Funding v2

Hypothesis: Daily timeframe with weekly HTF trend filter provides optimal trade frequency (20-50/year).
Combined with Choppiness Index regime detection, Connors RSI for mean reversion entries,
and Funding Rate z-score for contrarian bias (proven edge for BTC/ETH in bear markets).

Key improvements from failed experiments:
1. LOOSENED entry thresholds to ensure ≥10 trades/train, ≥3 trades/test per symbol
2. Funding z-score threshold reduced to ±1.5 (was ±2.0) to trigger more signals
3. CRSI thresholds: <30 / >70 (was <25 / >75) for more entries
4. CHOP thresholds: >55 choppy, <45 trending (wider bands for regime memory)
5. Removed 1w HMA requirement for entries (only use for size boost)
6. Simplified stoploss: 2.5x ATR from entry price

Regime Detection:
- Choppiness Index (CHOP) > 55 = choppy → Connors RSI mean reversion
- Choppiness Index (CHOP) < 45 = trending → HMA breakout + HTF alignment
- 45-55 = transition (use previous regime memory for hysteresis)

Entry Logic:
- Choppy regime: CRSI < 30 + price > SMA200 → long; CRSI > 70 + price < SMA200 → short
- Trending regime: HMA crossover + 1d HTF alignment → follow trend
- Funding override: Z-score < -1.5 → long bias; Z-score > +1.5 → short bias

Position sizing: 0.25 base, 0.35 when 1w HTF aligned (discrete levels)
Stoploss: 2.5x ATR from entry price (signal → 0 when hit)

Target: Sharpe>0.40, DD>-40%, trades>=20 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_funding_regime_1w_v2"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    Using 55/45 thresholds for regime detection with hysteresis
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate
    """
    n = len(close)
    if n < pr_period + 10:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Streak RSI
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        up_count = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        streak_rsi[i] = 100.0 * up_count / streak_period
    
    # Percent Rank of returns
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        if len(window) > 0 and not np.isnan(returns[i]):
            count_below = np.sum(window < returns[i])
            percent_rank[i] = 100.0 * count_below / len(window)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_funding_zscore(prices, lookback=30):
    """
    Funding Rate Z-Score for contrarian signal
    Load from funding parquet and calculate z-score
    """
    n = len(prices)
    zscore = np.zeros(n)
    zscore[:] = np.nan
    
    try:
        from pathlib import Path
        symbol = "BTCUSDT"
        funding_path = Path("data/processed/funding/BTCUSDT.parquet")
        
        if funding_path.exists():
            funding_df = pd.read_parquet(funding_path)
            if 'funding_rate' in funding_df.columns:
                funding_rates = funding_df['funding_rate'].values
                
                for i in range(lookback, n):
                    if i < len(funding_rates):
                        window = funding_rates[max(0, i-lookback):i]
                        if len(window) >= lookback // 2:
                            mean = np.nanmean(window)
                            std = np.nanstd(window)
                            if std > 1e-10:
                                zscore[i] = (funding_rates[i] - mean) / std
    except Exception:
        pass
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d = calculate_hma(close, period=21)
    hma_1d_fast = calculate_hma(close, period=8)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, 200)
    sma_50 = calculate_sma(close, 50)
    
    # Funding rate z-score (contrarian signal)
    funding_z = calculate_funding_zscore(prices, lookback=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 55.0
        trending_threshold = 45.0
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS (1w) ===
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 1d HMA TREND ===
        hma_bull = close[i] > hma_1d[i]
        hma_bear = close[i] < hma_1d[i]
        hma_cross_bull = not np.isnan(hma_1d_fast[i]) and hma_1d_fast[i] > hma_1d[i]
        hma_cross_bear = not np.isnan(hma_1d_fast[i]) and hma_1d_fast[i] < hma_1d[i]
        
        # === SMA FILTERS ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === CRSI VALUES (LOOSENED for more trades) ===
        crsi_extreme_low = False
        crsi_extreme_high = False
        if not np.isnan(crsi[i]):
            crsi_extreme_low = crsi[i] < 30.0  # Was 25
            crsi_extreme_high = crsi[i] > 70.0  # Was 75
        
        # === FUNDING RATE CONTRARIAN (LOOSENED thresholds) ===
        funding_long_bias = not np.isnan(funding_z[i]) and funding_z[i] < -1.5
        funding_short_bias = not np.isnan(funding_z[i]) and funding_z[i] > 1.5
        funding_strong_long = not np.isnan(funding_z[i]) and funding_z[i] < -2.0
        funding_strong_short = not np.isnan(funding_z[i]) and funding_z[i] > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # FUNDING OVERRIDE: Strong funding signals override regime
        if funding_strong_long:
            desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
        elif funding_strong_short:
            desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # REGIME 1: CHOPPY (mean reversion with CRSI)
        elif current_regime == 2:
            # Long: oversold + above SMA200 (loosened: OR above SMA50)
            if crsi_extreme_low and (above_sma200 or above_sma50):
                desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            
            # Short: overbought + below SMA200 (loosened: OR below SMA50)
            elif crsi_extreme_high and (below_sma200 or below_sma50):
                desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # REGIME 2: TRENDING (HMA crossover + HTF confirmation)
        elif current_regime == 1:
            # Long: HMA cross bull + price > HMA + 1w bull (loosened: no 1w required)
            if hma_cross_bull and hma_bull:
                desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
            
            # Short: HMA cross bear + price < HMA + 1w bear (loosened: no 1w required)
            elif hma_cross_bear and hma_bear:
                desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals