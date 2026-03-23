#!/usr/bin/env python3
"""
Experiment #996: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + Funding Contrarian

Hypothesis: After 995 failed experiments, combining Connors RSI (proven 75% win rate) with
Choppiness Index regime filtering and Funding Rate contrarian signals should work across
ALL symbols (BTC/ETH/SOL) on 12h timeframe with minimal fee drag.

Why this should work:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > SMA200 (75% win rate in research)
   - Short: CRSI > 90 + price < SMA200
   - Works in both bull and bear markets (mean reversion)

2. Choppiness Index regime filter:
   - CHOP(14) > 61.8 = range (use mean reversion/CRSI)
   - CHOP < 38.2 = trend (use trend following)
   - Avoids whipsaw in unclear regimes

3. Funding Rate Contrarian (BTC/ETH specific edge):
   - Z-score(funding, 30d) > +2 → short (crowded longs)
   - Z-score < -2 → long (crowded shorts)
   - Reported Sharpe 0.8-1.5 through 2022 crash

4. 1d HMA(21) for macro trend bias:
   - Only long if price > 1d HMA (bullish macro)
   - Only short if price < 1d HMA (bearish macro)

5. 12h timeframe benefits:
   - Target 20-50 trades/year (low fee drag ~1-2.5%)
   - Clearer signals than lower TF
   - HTF (1d) provides strong trend bias

Critical improvements from failures:
- RELAXED CRSI thresholds (15/85 not 10/90) to ensure >= 30 trades
- Funding as confluence (not sole signal) to avoid 0-trade strategies
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Proper stoploss (2.5x ATR) via signal → 0

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_funding_1d_hma_regime_atr_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Percentile rank of price change over 100 periods
    
    Long signal: CRSI < 15 (oversold)
    Short signal: CRSI > 85 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    streak[0] = 1
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = -(abs(streak[i-1]) + 1)
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = calculate_rsi(streak, period=streak_period)
    
    # Component 3: PercentRank of price change
    pct_change = np.zeros(n)
    pct_change[0] = 0
    for i in range(1, n):
        if close[i-1] > 0:
            pct_change[i] = (close[i] - close[i-1]) / close[i-1] * 100
        else:
            pct_change[i] = 0
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = pct_change[i-rank_period+1:i+1]
        current = pct_change[i]
        rank = np.sum(window < current)
        percent_rank[i] = rank / rank_period * 100
    
    # Combine components
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
    """
    Choppiness Index — measures market choppy vs trending.
    
    CHOP > 61.8 = Range/Ranging (use mean reversion)
    CHOP < 38.2 = Trending (use trend following)
    38.2 <= CHOP <= 61.8 = Neutral/Transition
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

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
        # Align funding to prices length
        if len(funding_rates) >= n:
            funding_rates = funding_rates[-n:]
        else:
            funding_rates = np.concatenate([np.zeros(n - len(funding_rates)), funding_rates])
    except:
        funding_rates = np.zeros(n)
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    sma_200_12h = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for macro regime
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
    
    for i in range(250, n):  # Need 200 for SMA + 100 for CRSI + buffer
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(sma_200_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(funding_z[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 61.8
        trending_regime = chop_12h[i] < 38.2
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 15
        crsi_overbought = crsi_12h[i] > 85
        crsi_extreme_oversold = crsi_12h[i] < 10
        crsi_extreme_overbought = crsi_12h[i] > 90
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200_12h[i]
        below_sma200 = close[i] < sma_200_12h[i]
        
        # === FUNDING RATE CONTRARIAN ===
        funding_extreme_short = funding_z[i] > 2.0  # Too many longs → short
        funding_extreme_long = funding_z[i] < -2.0  # Too many shorts → long
        funding_moderate_short = funding_z[i] > 1.0
        funding_moderate_long = funding_z[i] < -1.0
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 61.8) — Mean Reversion with CRSI ===
        if ranging_regime:
            # Long: CRSI oversold + above SMA200 + macro support
            if crsi_oversold and above_sma200:
                if macro_bull:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            # Long: CRSI extreme oversold (relaxed for more trades)
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            # Long: Funding extreme long contrarian
            elif funding_extreme_long:
                if macro_bull or above_sma200:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + below SMA200 + macro support
            if crsi_overbought and below_sma200:
                if macro_bear:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
            # Short: CRSI extreme overbought
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            # Short: Funding extreme short contrarian
            elif funding_extreme_short:
                if macro_bear or below_sma200:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 38.2) — Trend Following ===
        elif trending_regime:
            # Long: Macro bull + CRSI pullback (not extreme)
            if macro_bull and above_sma200:
                if crsi_12h[i] < 40:  # Pullback in uptrend
                    desired_signal = BASE_SIZE
                elif funding_moderate_long:
                    desired_signal = REDUCED_SIZE
            
            # Short: Macro bear + CRSI rally (not extreme)
            if macro_bear and below_sma200:
                if crsi_12h[i] > 60:  # Rally in downtrend
                    desired_signal = -BASE_SIZE
                elif funding_moderate_short:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: Funding contrarian + trend confluence
            if funding_extreme_long:
                if macro_bull or above_sma200:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            
            if funding_extreme_short:
                if macro_bear or below_sma200:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
            
            # Secondary: CRSI mean reversion (relaxed thresholds)
            if crsi_12h[i] < 20 and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if crsi_12h[i] > 80 and desired_signal == 0:
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
                # Hold long if macro intact and CRSI not overbought
                if (macro_bull or above_sma200) and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro intact and CRSI not oversold
                if (macro_bear or below_sma200) and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses + CRSI overbought
            if macro_bear and below_sma200 and crsi_12h[i] > 80:
                desired_signal = 0.0
            # Exit if funding flips extreme short
            if funding_extreme_short and crsi_12h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses + CRSI oversold
            if macro_bull and above_sma200 and crsi_12h[i] < 20:
                desired_signal = 0.0
            # Exit if funding flips extreme long
            if funding_extreme_long and crsi_12h[i] < 30:
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