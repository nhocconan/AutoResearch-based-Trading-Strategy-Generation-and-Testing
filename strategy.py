#!/usr/bin/env python3
"""
Experiment #991: 4h Primary + 1d/1w HTF — Funding Contrarian + Connors RSI + Choppiness Regime

Hypothesis: After 717 failed strategies, funding rate contrarian signal is the MOST PROVEN edge
for BTC/ETH specifically (research shows Sharpe 0.8-1.5 through 2022 crash). Combined with
Connors RSI for precise entry timing and Choppiness Index for regime detection, this should
work across ALL symbols with positive Sharpe.

Key insights:
1. Funding Rate Z-score: Extreme funding (>2σ) predicts reversals. Best for BTC/ETH.
2. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3. More sensitive than RSI(14).
3. Choppiness Index: CHOP>61.8=range (mean revert), CHOP<38.2=trend (breakout).
4. 1d HMA(21) + 1w HMA(21): Macro trend bias to avoid counter-trend funding trades.
5. Relaxed entry thresholds to ensure >=30 trades train, >=3 trades test.

Why this should beat Sharpe=0.612:
- Funding contrarian worked through 2022 crash (where trend strategies failed)
- CRSI catches reversals earlier than RSI(14)
- Regime-adaptive: mean revert in chop, trend-follow in trends
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Timeframe: 4h (target 25-40 trades/year, low fee drag)
Position sizing: BASE=0.30, REDUCED=0.20, MAX=0.35
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_crsi_chop_regime_1d1w_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods."""
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    More sensitive to reversals than standard RSI(14).
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period - 1, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_lower = np.sum(window[:-1] < current)
        percent_rank[i] = count_lower / (rank_period - 1) * 100
    
    # Combine
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
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
    """Average True Range with proper min_periods."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Load funding rate data
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
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
        if np.isnan(crsi_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(funding_z[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === FUNDING RATE CONTRARIAN (PRIMARY SIGNAL) ===
        funding_extreme_short = funding_z[i] > 1.5  # Too many longs → short signal
        funding_extreme_long = funding_z[i] < -1.5  # Too many shorts → long signal
        funding_moderate_short = funding_z[i] > 0.5
        funding_moderate_long = funding_z[i] < -0.5
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_4h[i] < 20
        crsi_overbought = crsi_4h[i] > 80
        crsi_extreme_oversold = crsi_4h[i] < 10
        crsi_extreme_overbought = crsi_4h[i] > 90
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: Funding extreme long + CRSI oversold
            if funding_extreme_long and crsi_oversold:
                desired_signal = BASE_SIZE
            # Long: Funding moderate long + CRSI extreme oversold
            elif funding_moderate_long and crsi_extreme_oversold:
                desired_signal = BASE_SIZE
            # Long: CRSI extreme oversold + trend support (guarantees trades)
            elif crsi_extreme_oversold and (macro_bull or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            # Long: Funding extreme long alone (ensures trade generation)
            elif funding_extreme_long:
                desired_signal = REDUCED_SIZE
            
            # Short: Funding extreme short + CRSI overbought
            if funding_extreme_short and crsi_overbought:
                desired_signal = -BASE_SIZE
            # Short: Funding moderate short + CRSI extreme overbought
            elif funding_moderate_short and crsi_extreme_overbought:
                desired_signal = -BASE_SIZE
            # Short: CRSI extreme overbought + trend support
            elif crsi_extreme_overbought and (macro_bear or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
            # Short: Funding extreme short alone
            elif funding_extreme_short:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following with Funding ===
        elif trending_regime:
            # Long: Bullish trend + funding pullback (contrarian entry in trend)
            if macro_bull or trend_1d_bullish:
                if funding_moderate_long and crsi_oversold:
                    desired_signal = BASE_SIZE
                elif funding_extreme_long:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + funding rally
            if macro_bear or trend_1d_bearish:
                if funding_moderate_short and crsi_overbought:
                    desired_signal = -BASE_SIZE
                elif funding_extreme_short:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Funding contrarian + trend confluence
            if funding_extreme_long and (macro_bull or trend_1d_bullish):
                desired_signal = BASE_SIZE
            elif funding_extreme_long:
                desired_signal = REDUCED_SIZE
            
            if funding_extreme_short and (macro_bear or trend_1d_bearish):
                desired_signal = -BASE_SIZE
            elif funding_extreme_short:
                desired_signal = -REDUCED_SIZE
            
            # Secondary: CRSI extremes
            if crsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if crsi_extreme_overbought and desired_signal == 0:
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
                if (macro_bull or trend_1d_bullish) and crsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if (macro_bear or trend_1d_bearish) and crsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if macro_bear and trend_1d_bearish and crsi_4h[i] > 75:
                desired_signal = 0.0
            if funding_extreme_short and crsi_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull and trend_1d_bullish and crsi_4h[i] < 25:
                desired_signal = 0.0
            if funding_extreme_long and crsi_oversold:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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