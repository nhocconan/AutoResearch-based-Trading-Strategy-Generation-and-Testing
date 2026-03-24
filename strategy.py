#!/usr/bin/env python3
"""
Experiment #321: 15m Primary + 1h/4h/1d HTF — Multi-Regime Adaptive with Funding Contrarian

Hypothesis: 15m timeframe needs LOOSE entry conditions to generate trades (learned from #309,#316,#317,#319 failures).
Combine 4h HMA trend direction + 1h RSI momentum + 15m Connors RSI entries + Funding rate contrarian overlay.
Key insight: Previous 15m strategies had TOO MANY filters causing 0 trades. This uses 2-3 confluence max.

Regime Detection:
- Choppiness Index > 55 = choppy → Connors RSI mean reversion (buy oversold, sell overbought)
- Choppiness Index < 45 = trending → HMA breakout + RSI momentum
- 45-55 = transition (use previous regime)

Entry Logic (LOOSENED for trade generation):
- Choppy: CRSI < 30 + 4h HMA bull → long; CRSI > 70 + 4h HMA bear → short
- Trending: 15m close > 4h HMA + 1h RSI > 50 → long; opposite for short
- Funding override: Z-score < -1.5 → long bias; > +1.5 → short bias

Position sizing: 0.15 base (15m needs smaller size due to higher frequency), 0.25 when HTF aligned
Stoploss: 2.0x ATR from entry (tighter for 15m)
Session filter: 00-14 UTC only (reduces trades to target 40-100/year)

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_crsi_funding_1h4h1d_v1"
timeframe = "15m"
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
    """
    n = len(prices)
    zscore = np.zeros(n)
    zscore[:] = np.nan
    
    try:
        from pathlib import Path
        symbol = "BTCUSDT"
        funding_path = Path("data/processed/funding/BTCUSDT.parquet")
        
        # Try different symbol paths
        if not funding_path.exists():
            funding_path = Path("data/processed/funding/ETHUSDT.parquet")
        if not funding_path.exists():
            funding_path = Path("data/processed/funding/SOLUSDT.parquet")
        
        if funding_path.exists():
            funding_df = pd.read_parquet(funding_path)
            if 'funding_rate' in funding_df.columns:
                funding_rates = funding_df['funding_rate'].values
                
                # Pad if funding is shorter than prices
                if len(funding_rates) < n:
                    funding_rates = np.concatenate([funding_rates, np.full(n - len(funding_rates), np.nan)])
                
                for i in range(lookback, n):
                    if i < len(funding_rates) and not np.isnan(funding_rates[i]):
                        window = funding_rates[max(0, i-lookback):i]
                        valid_window = window[~np.isnan(window)]
                        if len(valid_window) >= lookback // 2:
                            mean = np.mean(valid_window)
                            std = np.std(valid_window)
                            if std > 1e-10:
                                zscore[i] = (funding_rates[i] - mean) / std
    except Exception:
        pass
    
    return zscore

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time for session filtering"""
    try:
        # Try to parse open_time as timestamp
        if 'open_time' in prices.columns:
            # Binance open_time is in milliseconds
            timestamps = prices['open_time'].values / 1000.0
            hours = np.array([(int(ts) // 3600) % 24 for ts in timestamps])
            return hours
    except Exception:
        pass
    
    # Fallback: assume uniform 15m bars starting from midnight
    n = len(prices)
    hours = np.zeros(n, dtype=int)
    for i in range(n):
        hours[i] = (i * 15 // 60) % 24  # 15min bars
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h RSI for momentum
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, 200)
    
    # Funding rate z-score (contrarian signal)
    funding_z = calculate_funding_zscore(prices, lookback=30)
    
    # Session hours for filtering
    hours = get_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-14 UTC only for London/NY overlap) ===
        in_session = hours[i] >= 0 and hours[i] <= 14
        
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
        
        # === HTF BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === 1h RSI MOMENTUM ===
        rsi_1h = rsi_1h_aligned[i]
        rsi_bull = not np.isnan(rsi_1h) and rsi_1h > 50.0
        rsi_bear = not np.isnan(rsi_1h) and rsi_1h < 50.0
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CRSI VALUES (LOOSENED for trade generation) ===
        crsi_extreme_low = False
        crsi_extreme_high = False
        if not np.isnan(crsi[i]):
            crsi_extreme_low = crsi[i] < 30.0  # Loosened from 20
            crsi_extreme_high = crsi[i] > 70.0  # Loosened from 80
        
        # === FUNDING RATE CONTRARIAN ===
        funding_long_bias = not np.isnan(funding_z[i]) and funding_z[i] < -1.5
        funding_short_bias = not np.isnan(funding_z[i]) and funding_z[i] > 1.5
        funding_strong_long = not np.isnan(funding_z[i]) and funding_z[i] < -2.0
        funding_strong_short = not np.isnan(funding_z[i]) and funding_z[i] > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # FUNDING OVERRIDE: Strong funding signals override regime
        if funding_strong_long:
            desired_signal = SIZE_STRONG
        elif funding_strong_short:
            desired_signal = -SIZE_STRONG
        
        # REGIME 1: CHOPPY (mean reversion with CRSI)
        elif current_regime == 2:
            # Long: oversold + 4h HMA bull OR funding long bias
            if crsi_extreme_low and (htf_4h_bull or funding_long_bias):
                desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            
            # Short: overbought + 4h HMA bear OR funding short bias
            elif crsi_extreme_high and (htf_4h_bear or funding_short_bias):
                desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
        
        # REGIME 2: TRENDING (momentum with HMA + RSI)
        elif current_regime == 1:
            # Long: 15m HMA bull + 1h RSI bull + 4h HMA bull
            if hma_bull and rsi_bull and htf_4h_bull:
                desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            
            # Short: 15m HMA bear + 1h RSI bear + 4h HMA bear
            elif hma_bear and rsi_bear and htf_4h_bear:
                desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
        
        # === SESSION FILTER APPLY ===
        if not in_session and desired_signal != 0.0:
            # Reduce position outside session hours
            if in_position:
                pass  # Keep existing position
            else:
                desired_signal = 0.0  # Don't enter new position
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals