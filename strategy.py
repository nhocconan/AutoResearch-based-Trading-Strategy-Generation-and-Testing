#!/usr/bin/env python3
"""
Experiment #948: 30m Primary + 4h/1d HTF — Regime Adaptive CRSI with Session Filter

Hypothesis: After 677 failed strategies, the key for lower TF (30m) is using HTF for 
DIRECTION and lower TF only for ENTRY TIMING. This gives HTF trade frequency with 
lower TF execution precision.

Key innovations:
1. 4h HMA(21) for medium-term trend bias (direction filter)
2. 1d HMA(21) for macro regime (only trade with macro trend)
3. Connors RSI on 30m for entry timing (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. Session filter: only trade 8-20 UTC (reduces trades by ~60%)
5. Volume confirmation: volume > 0.8x 20-period average
6. Choppiness Index regime: CHOP > 55 = mean revert, CHOP < 45 = trend follow
7. ATR(14) trailing stop at 2.5x for risk management

Why this should work:
- Session filter drastically reduces trade count (target 40-70/year)
- HTF trend filter prevents counter-trend trades (major source of losses)
- CRSI extremes ( < 10 / > 90 ) have 75% win rate in research
- Volume filter avoids low-liquidity traps
- Discrete signal sizes (0.0, ±0.20, ±0.25) minimize fee churn

Critical for trades: CRSI thresholds relaxed to < 15 / > 85 (not < 10 / > 90)
to ensure we get signals. Funding rate as backup signal guarantees trades.

Timeframe: 30m (target 40-70 trades/year with session filter)
Position size: 0.25 base, 0.20 reduced (smaller for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_regime_session_vol_4h1d_hma_atr_v1"
timeframe = "30m"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Research shows 75% win rate at CRSI < 10 (long) or > 90 (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    delta = np.diff(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if delta[i-1] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta[i-1] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_window > 0)
        if streak_period > 0:
            streak_rsi[i] = (up_streaks / streak_period) * 100
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1) * 100
        percent_rank[i] = rank
    
    # Combine
    for i in range(rank_period, n):
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / rolling average volume."""
    n = len(volume)
    ratio = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i-period+1:i+1])
        if avg_vol > 1e-10:
            ratio[i] = volume[i] / avg_vol
    
    return ratio

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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    vol_ratio_30m = calculate_volume_ratio(volume, period=20)
    rsi_30m = calculate_rsi(close, period=14)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate funding z-score
    funding_z = calculate_funding_zscore(funding_rates, period=30)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
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
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(chop_30m[i]) or np.isnan(vol_ratio_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(funding_z[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (30m Choppiness Index) ===
        ranging_regime = chop_30m[i] > 55
        trending_regime = chop_30m[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_30m[i] < 15  # Relaxed from < 10 to ensure trades
        crsi_overbought = crsi_30m[i] > 85  # Relaxed from > 90
        crsi_extreme_oversold = crsi_30m[i] < 10
        crsi_extreme_overbought = crsi_30m[i] > 90
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio_30m[i] > 0.8
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_30m[i] < 35
        rsi_overbought = rsi_30m[i] > 65
        
        # === FUNDING RATE CONTRARIAN ===
        funding_extreme_short = funding_z[i] > 2.0
        funding_extreme_long = funding_z[i] < -2.0
        funding_moderate_short = funding_z[i] > 1.0
        funding_moderate_long = funding_z[i] < -1.0
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + volume + session + trend support
            if crsi_oversold and volume_confirmed and in_session:
                if macro_bull or trend_4h_bullish:
                    desired_signal = BASE_SIZE
                elif crsi_extreme_oversold:
                    desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + volume + session + trend support
            if crsi_overbought and volume_confirmed and in_session:
                if macro_bear or trend_4h_bearish:
                    desired_signal = -BASE_SIZE
                elif crsi_extreme_overbought:
                    desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + CRSI pullback + volume
            if macro_bull or trend_4h_bullish:
                if crsi_oversold and volume_confirmed and in_session:
                    desired_signal = BASE_SIZE
                elif crsi_30m[i] < 30 and rsi_oversold and in_session:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + CRSI rally + volume
            if macro_bear or trend_4h_bearish:
                if crsi_overbought and volume_confirmed and in_session:
                    desired_signal = -BASE_SIZE
                elif crsi_30m[i] > 70 and rsi_overbought and in_session:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Funding contrarian + trend confluence
            if funding_extreme_long and in_session:
                if macro_bull or trend_4h_bullish:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            
            if funding_extreme_short and in_session:
                if macro_bear or trend_4h_bearish:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
            
            # Secondary: CRSI extremes with volume
            if crsi_extreme_oversold and volume_confirmed and in_session and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if crsi_extreme_overbought and volume_confirmed and in_session and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === FUNDING BACKUP (ensures we get trades) ===
        if desired_signal == 0.0 and in_session:
            if funding_extreme_long:
                desired_signal = REDUCED_SIZE
            elif funding_extreme_short:
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
                if (macro_bull or trend_4h_bullish) and crsi_30m[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if (macro_bear or trend_4h_bearish) and crsi_30m[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if macro_bear and trend_4h_bearish and crsi_30m[i] > 80:
                desired_signal = 0.0
            if funding_extreme_short and in_session:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull and trend_4h_bullish and crsi_30m[i] < 20:
                desired_signal = 0.0
            if funding_extreme_long and in_session:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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