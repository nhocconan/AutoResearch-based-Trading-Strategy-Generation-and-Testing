#!/usr/bin/env python3
"""
Experiment #966: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + Funding Contrarian

Hypothesis: After 664 failed strategies, combining Connors RSI (proven 75% win rate) with
Choppiness Index regime detection and funding rate contrarian signals should work on 12h
timeframe across ALL symbols (BTC/ETH/SOL).

Why 12h timeframe:
- Target 20-50 trades/year (minimal fee drag)
- HTF signals (1d) provide strong macro trend bias
- Less noise than 4h/1h, clearer regime detection
- Proven to work in both bull and bear markets

Key innovations:
1. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More sensitive than regular RSI, catches reversals faster
   - Long: CRSI < 10, Short: CRSI > 90
2. CHOPPINESS INDEX regime: CHOP(14) > 55 = range (mean revert), CHOP < 45 = trend
3. 1d HMA(21) for macro trend bias (only trade with macro trend in trending regime)
4. Funding rate contrarian as additional confluence (Z-score > 2 or < -2)
5. ATR(14) trailing stoploss at 2.5x

Position sizing:
- BASE_SIZE = 0.30 (30% of capital)
- REDUCED_SIZE = 0.20 (20% of capital)
- Discrete levels minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_connors_rsi_chop_regime_1d_hma_funding_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """RSI Streak Component of Connors RSI.
    Measures consecutive up/down days."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like score (0-100)
    abs_streak = np.abs(streak)
    max_streak = np.max(abs_streak[~np.isnan(abs_streak)]) if np.any(~np.isnan(abs_streak)) else 1
    
    # Normalize: longer streak = more extreme score
    for i in range(n):
        if np.isnan(streak[i]):
            continue
        if streak[i] > 0:
            streak_rsi[i] = 50 + (streak[i] / max(max_streak, 1)) * 50
        elif streak[i] < 0:
            streak_rsi[i] = 50 - (abs(streak[i]) / max(max_streak, 1)) * 50
        else:
            streak_rsi[i] = 50
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank Component of Connors RSI.
    Measures where current price change ranks vs last N periods."""
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period + 1:
        return pr
    
    price_change = np.diff(close)
    price_change = np.concatenate([[0], price_change])
    
    for i in range(period, n):
        window = price_change[i-period+1:i+1]
        current = price_change[i]
        rank = np.sum(window < current)
        pr[i] = (rank / period) * 100
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_3 = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    pr = calculate_percent_rank(close, period=pr_period)
    
    n = len(close)
    crsi = np.full(n, np.nan)
    
    for i in range(n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + pr[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_funding_zscore(funding_series, period=30):
    """Z-score of funding rate over lookback period."""
    n = len(funding_series)
    zscore = np.full(n, np.nan)
    
    if n < period:
        return zscore
    
    for i in range(period - 1, n):
        window = funding_series[i-period+1:i+1]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        if std > 1e-10:
            zscore[i] = (funding_series[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Load funding rate data if available
    symbol = prices['symbol'].iloc[0] if 'symbol' in prices.columns else 'BTCUSDT'
    funding_path = f"data/processed/funding/{symbol}.parquet"
    try:
        df_funding = pd.read_parquet(funding_path)
        funding_rates = df_funding['funding_rate'].values
        if len(funding_rates) >= n:
            funding_rates = funding_rates[-n:]
        else:
            funding_rates = np.concatenate([np.zeros(n - len(funding_rates)), funding_rates])
    except:
        funding_rates = np.zeros(n)
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate funding z-score
    funding_z = calculate_funding_zscore(funding_rates, period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(funding_z[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_low = crsi_12h[i] < 10
        crsi_extreme_high = crsi_12h[i] > 90
        crsi_low = crsi_12h[i] < 20
        crsi_high = crsi_12h[i] > 80
        
        # === FUNDING RATE CONTRARIAN ===
        funding_extreme_short = funding_z[i] > 2.0
        funding_extreme_long = funding_z[i] < -2.0
        funding_moderate_short = funding_z[i] > 1.0
        funding_moderate_long = funding_z[i] < -1.0
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI extreme low (strong mean reversion signal)
            if crsi_extreme_low:
                desired_signal = BASE_SIZE
            # Long: CRSI low + funding extreme long (contrarian confluence)
            elif crsi_low and funding_extreme_long:
                desired_signal = BASE_SIZE
            # Long: Funding extreme long alone (ensures trades)
            elif funding_extreme_long:
                desired_signal = REDUCED_SIZE
            # Long: CRSI low + macro support
            elif crsi_low and macro_bull:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI extreme high
            if crsi_extreme_high:
                desired_signal = -BASE_SIZE
            # Short: CRSI high + funding extreme short
            elif crsi_high and funding_extreme_short:
                desired_signal = -BASE_SIZE
            # Short: Funding extreme short alone
            elif funding_extreme_short:
                desired_signal = -REDUCED_SIZE
            # Short: CRSI high + macro resistance
            elif crsi_high and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following with Pullbacks ===
        elif trending_regime:
            # Long: Macro bull + CRSI pullback (buy dip in uptrend)
            if macro_bull:
                if crsi_low:
                    desired_signal = BASE_SIZE
                elif crsi_extreme_low:
                    desired_signal = BASE_SIZE
                elif funding_moderate_long:
                    desired_signal = REDUCED_SIZE
            
            # Short: Macro bear + CRSI rally (sell rip in downtrend)
            if macro_bear:
                if crsi_high:
                    desired_signal = -BASE_SIZE
                elif crsi_extreme_high:
                    desired_signal = -BASE_SIZE
                elif funding_moderate_short:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Funding contrarian only
            if funding_extreme_long:
                desired_signal = BASE_SIZE
            elif funding_moderate_long and crsi_low:
                desired_signal = REDUCED_SIZE
            
            if funding_extreme_short:
                desired_signal = -BASE_SIZE
            elif funding_moderate_short and crsi_high:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bull and CRSI not extreme high
                if macro_bull and crsi_12h[i] < 85:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bear and CRSI not extreme low
                if macro_bear and crsi_12h[i] > 15:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses + CRSI extreme high
            if macro_bear and crsi_extreme_high:
                desired_signal = 0.0
            # Exit if funding flips extreme short
            if funding_extreme_short:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses + CRSI extreme low
            if macro_bull and crsi_extreme_low:
                desired_signal = 0.0
            # Exit if funding flips extreme long
            if funding_extreme_long:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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
        
        signals[i] = desired_signal
    
    return signals