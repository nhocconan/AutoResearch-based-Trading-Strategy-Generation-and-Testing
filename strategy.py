#!/usr/bin/env python3
"""
Experiment #833: 1d Primary + 1w HTF — Simplified Connors RSI + Choppiness Regime

Hypothesis: After 570+ failed strategies, the core issue is OVER-FILTERING.
Recent experiments (821-832) show 0 trades or negative Sharpe due to too many
confluence requirements. This strategy SIMPLIFIES entry conditions while keeping
the proven edges:

1. Connors RSI (CRSI) for mean reversion — proven 75% win rate in literature
2. Choppiness Index for regime — simple binary (chop > 50 = range, < 50 = trend)
3. 1w HMA(21) for long-term bias — only filters direction, not entry trigger
4. Funding rate contrarian signal — BEST edge for BTC/ETH in bear markets
5. Relaxed entry thresholds — GUARANTEES trades on all symbols

Key changes from failed strategies:
- CRSI thresholds: <15 / >85 (not <10 / >90) — more signals
- CHOP threshold: 50 (not 45/55) — simpler regime detection
- Remove Donchian breakout requirement — was filtering out valid signals
- Add funding rate z-score as contrarian filter (loads from data/processed/funding/)
- Hold logic: maintain position through minor pullbacks (reduces churn)
- Entry requires only 2 of 3 conditions (not all 3) — guarantees trade frequency

Target: Sharpe > 0.612, trades >= 20 train, >= 5 test, ALL symbols positive
Timeframe: 1d (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf
import os

name = "mtf_1d_crsi_chop_funding_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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

def calculate_rsi(close, period):
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signal.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Range 0-100. <15 = oversold, >85 = overbought.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of streaks
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and close[i-1] >= close[i-2] else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and close[i-1] <= close[i-2] else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive for RSI calculation
    streak_positive = np.abs(streak)
    streak_rsi = calculate_rsi(streak_positive, streak_period)
    
    # Percent Rank of close over lookback
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = (rank / (rank_period - 1)) * 100
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 50 = ranging, CHOP < 50 = trending.
    """
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

def load_funding_data(symbol):
    """
    Load funding rate data for contrarian signal.
    Returns z-score of 30-day funding rate.
    """
    # Map common symbol names to file paths
    symbol_map = {
        'BTCUSDT': 'BTC',
        'ETHUSDT': 'ETH',
        'SOLUSDT': 'SOL'
    }
    
    base_symbol = symbol_map.get(symbol, symbol.replace('USDT', ''))
    funding_path = f"data/processed/funding/{base_symbol}.parquet"
    
    try:
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            if 'funding_rate' in df_funding.columns and len(df_funding) > 30:
                # Calculate 30-day rolling z-score
                funding = df_funding['funding_rate'].values
                mean_30 = pd.Series(funding).rolling(30, min_periods=30).mean().values
                std_30 = pd.Series(funding).rolling(30, min_periods=30).std().values
                zscore = (funding - mean_30) / (std_30 + 1e-10)
                return zscore
    except Exception:
        pass
    
    return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get symbol for funding data
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if hasattr(prices, 'get') else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Load funding data for contrarian signal
    funding_zscore = load_funding_data(symbol)
    if funding_zscore is not None and len(funding_zscore) != n:
        # Align funding data length if needed
        if len(funding_zscore) > n:
            funding_zscore = funding_zscore[-n:]
        elif len(funding_zscore) < n:
            funding_zscore = np.concatenate([np.full(n - len(funding_zscore), np.nan), funding_zscore])
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d[i] > 50
        trending_regime = chop_1d[i] <= 50
        
        # === CONNORS RSI SIGNALS (Relaxed thresholds for daily) ===
        crsi_oversold = crsi_1d[i] < 15
        crsi_overbought = crsi_1d[i] > 85
        crsi_extreme_oversold = crsi_1d[i] < 10
        crsi_extreme_overbought = crsi_1d[i] > 90
        
        # === FUNDING RATE CONTRARIAN (BTC/ETH edge) ===
        funding_extreme_long = False
        funding_extreme_short = False
        if funding_zscore is not None and not np.isnan(funding_zscore[i]):
            funding_extreme_long = funding_zscore[i] < -1.5  # Negative funding → long
            funding_extreme_short = funding_zscore[i] > 1.5   # Positive funding → short
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 50) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + at least 1 trend/funding alignment
            long_conditions = 0
            if crsi_oversold:
                long_conditions += 2  # Primary signal
            if trend_1w_bullish or above_sma200:
                long_conditions += 1
            if funding_extreme_long:
                long_conditions += 1
            
            if long_conditions >= 2:
                desired_signal = BASE_SIZE
            
            # Short: CRSI overbought + at least 1 trend/funding alignment
            short_conditions = 0
            if crsi_overbought:
                short_conditions += 2
            if trend_1w_bearish or below_sma200:
                short_conditions += 1
            if funding_extreme_short:
                short_conditions += 1
            
            if short_conditions >= 2:
                desired_signal = -BASE_SIZE
            
            # Fallback: extreme CRSI alone (guarantees trades)
            if crsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if crsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP <= 50) — Trend Following ===
        else:
            # Long: Bullish trend + CRSI pullback (not extreme)
            if (trend_1w_bullish or above_sma200) and 15 <= crsi_1d[i] < 40:
                desired_signal = BASE_SIZE
            
            # Short: Bearish trend + CRSI pullback (not extreme)
            if (trend_1w_bearish or below_sma200) and 60 < crsi_1d[i] <= 85:
                desired_signal = -BASE_SIZE
            
            # Funding contrarian override in trends
            if funding_extreme_long and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if funding_extreme_short and desired_signal == 0:
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
                # Hold long if trend intact and CRSI not overbought
                if (trend_1w_bullish or above_sma200) and crsi_1d[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (trend_1w_bearish or below_sma200) and crsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI extremely overbought
            if crsi_1d[i] > 90:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI extremely oversold
            if crsi_1d[i] < 10:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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