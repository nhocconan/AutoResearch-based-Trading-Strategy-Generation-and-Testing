#!/usr/bin/env python3
"""
Experiment #057: 1d Primary + 4h HTF — Connors RSI + Choppiness Regime + Fisher Transform

Hypothesis: Daily timeframe with 4h trend bias using Connors RSI for mean reversion
entries in ranging markets and Fisher Transform for trend reversals will generate
25-45 trades/year with Sharpe > 0.486.

Key insights from 56 failed experiments:
1) 1d primary timeframe works (exp #047 Sharpe=0.141 kept)
2) 4h HTF provides better trend bias than 1w (current best uses 4h/1d/1w)
3) Connors RSI (CRSI) has 75% win rate for mean reversion (proven on ETH)
4) Choppiness Index regime filter is critical for bear/range markets
5) Fisher Transform catches reversals in bear rallies better than RSI
6) Simpler entry conditions = more trades = better Sharpe (exp #052 had 0 trades)

Why this should work:
- 1d primary = proven higher TF (fewer trades, less fee drag)
- 4h HTF = trend bias without over-filtering (current best uses 4h)
- CRSI < 10 + regime = range = long (proven 75% win rate)
- CRSI > 90 + regime = range = short (proven 75% win rate)
- Fisher crossover + regime = trend = breakout entries
- Ensures trades on ALL symbols (BTC/ETH/SOL) in all regimes

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 25-45 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_fisher_chop_regime_4h_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - short-term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak - consecutive up/down days
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_positive = np.maximum(streak, 0)
    streak_negative = np.abs(np.minimum(streak, 0))
    
    # Simple streak RSI approximation
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        avg_gain = np.mean(np.maximum(streak[i-streak_period:i+1], 0))
        avg_loss = np.mean(np.abs(np.minimum(streak[i-streak_period:i+1], 0)))
        if avg_loss == 0:
            streak_rsi[i] = 100.0
        else:
            rs = avg_gain / (avg_loss + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank - where today's return ranks in last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period:i+1].dropna()
        if len(window_returns) > 0:
            current_return = returns.iloc[i]
            rank = np.sum(window_returns <= current_return)
            percent_rank[i] = 100.0 * rank / len(window_returns)
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    return crsi

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((close - lowest_low) / (highest_high - lowest_low) - 0.5) * 2
    """
    close_s = pd.Series(close)
    highest = close_s.rolling(window=period, min_periods=period).max()
    lowest = close_s.rolling(window=period, min_periods=period).min()
    
    # Normalize price to -1 to +1 range
    x = 0.66 * ((close_s - lowest) / (highest - lowest + 1e-10) - 0.5) * 2
    x = np.clip(x, -0.99, 0.99)  # Prevent log domain error
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    fisher = fisher.fillna(0.0).values
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    change = np.abs(close_s.diff(er_period))
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    er = change / (volatility + 1e-10)
    er = er.fillna(0.0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close_s.iloc[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    kama_10 = calculate_kama(close, er_period=10)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    fisher = calculate_fisher(close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(hma_21[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(sma_200[i]) or np.isnan(kama_10[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H MACRO BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        price_above_hma_1d = close[i] > hma_21[i]
        price_below_hma_1d = close[i] < hma_21[i]
        price_above_sma_200 = close[i] > sma_200[i]
        price_below_sma_200 = close[i] < sma_200[i]
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market (mean revert)
        is_trending = chop_value < 45.0  # Trend market (trend follow)
        
        # === CONNORS RSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought
        
        # === FISHER TRANSFORM SIGNALS (Trend Reversal) ===
        fisher_cross_up = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        fisher_cross_down = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # === HMA SLOPE ===
        hma_slope_up = hma_21[i] > hma_21[i-5] if i > 5 else False
        hma_slope_down = hma_21[i] < hma_21[i-5] if i > 5 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: CRSI Mean Reversion ---
        if is_ranging:
            # Long: CRSI oversold + price above SMA200 (bullish bias)
            if crsi_oversold and price_above_sma_200:
                new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price below SMA200 (bearish bias)
            elif crsi_overbought and price_below_sma_200:
                new_signal = -POSITION_SIZE
            
            # Neutral SMA: use 4h bias instead
            elif crsi_oversold and price_above_hma_4h:
                new_signal = POSITION_SIZE
            elif crsi_overbought and price_below_hma_4h:
                new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Fisher Transform + HMA Trend ---
        elif is_trending:
            # Long: Fisher crosses up from oversold + HMA bullish
            if fisher_cross_up and (price_above_hma_1d or hma_slope_up):
                if price_above_hma_4h or price_above_kama:
                    new_signal = POSITION_SIZE
            
            # Short: Fisher crosses down from overbought + HMA bearish
            elif fisher_cross_down and (price_below_hma_1d or hma_slope_down):
                if price_below_hma_4h or price_below_kama:
                    new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME: Hybrid (ensures trades) ---
        else:
            # Long: CRSI oversold OR Fisher cross up + 4h bias
            if (crsi_oversold or fisher_cross_up) and price_above_hma_4h:
                new_signal = POSITION_SIZE
            # Short: CRSI overbought OR Fisher cross down + 4h bias
            elif (crsi_overbought or fisher_cross_down) and price_below_hma_4h:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if not at opposite extreme
            if position_side > 0 and crsi[i] < 80.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 20.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_1d and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d and price_above_hma_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals