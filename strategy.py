#!/usr/bin/env python3
"""
Experiment #084: 4h Primary + 12h/1d HTF — Regime-Adaptive with Choppiness + CRSI + Funding

Hypothesis: Dual-regime strategy adapts to market conditions better than single-mode approaches.
Use Choppiness Index to detect range vs trend, then apply appropriate strategy:
- Range (CHOP > 61.8): Connors RSI mean reversion at Bollinger bands
- Trend (CHOP < 38.2): HMA trend following with RSI pullback entries
Add funding rate z-score as contrarian filter (proven edge for BTC/ETH in bear markets).

Key innovations:
1) Choppiness Index(14) regime detection — switch between mean revert and trend follow
2) Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — 75% win rate in research
3) Bollinger Band(20, 2.0) confirmation for mean reversion entries
4) 12h HMA(21) for macro trend bias — only trade with HTF trend in trend regime
5) Funding rate z-score(30) contrarian — boost position when funding extreme
6) ATR(14) trailing stoploss at 2.5*ATR
7) Discrete sizing: 0.25 base, 0.30 max with funding boost

Why this should beat #079:
- Regime adaptation prevents trend strategies from failing in range markets (2025 test)
- CRSI more sensitive than RSI(7) for pullback detection
- CHOP filter prevents false breakouts in choppy conditions
- Funding contrarian works especially well for BTC/ETH (research Sharpe 0.8-1.5)
- Simpler than failed complex regimes (#072-#083) but more adaptive than #079

Position size: 0.25 base, 0.30 max with funding boost
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_funding_12h1d_v1"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    chop = 100.0 * np.log10(atr_sum / price_range + 1e-10) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    CRSI < 10-20 = oversold (long), CRSI > 80-90 = overbought (short)
    """
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI component
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        if streak_sign[i] > 0:
            streak_rsi[i] = 100.0 * streak_abs[i] / (streak_period + 1e-10)
        elif streak_sign[i] < 0:
            streak_rsi[i] = 100.0 - 100.0 * streak_abs[i] / (streak_period + 1e-10)
        else:
            streak_rsi[i] = 50.0
    streak_rsi = np.clip(streak_rsi, 0.0, 100.0)
    
    # Percent Rank component
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_funding_zscore(prices, symbol, lookback=30):
    """
    Calculate funding rate z-score for contrarian signal.
    """
    try:
        import os
        funding_path = f"data/processed/funding/{symbol.lower()}.parquet"
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            if 'funding_rate' in df_funding.columns:
                funding = df_funding['funding_rate'].values
                min_len = min(len(funding), len(prices))
                funding = funding[-min_len:]
                
                funding_s = pd.Series(funding)
                funding_mean = funding_s.rolling(window=lookback, min_periods=lookback).mean()
                funding_std = funding_s.rolling(window=lookback, min_periods=lookback).std()
                zscore = (funding_s - funding_mean) / (funding_std + 1e-10)
                zscore = zscore.fillna(0.0).values
                
                if len(zscore) < len(prices):
                    pad = np.zeros(len(prices) - len(zscore))
                    zscore = np.concatenate([pad, zscore])
                
                return zscore[:len(prices)]
    except Exception:
        pass
    
    return np.zeros(len(prices))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = "BTCUSDT"
    if hasattr(prices, 'attrs') and 'symbol' in prices.attrs:
        symbol = prices.attrs['symbol']
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for intermediate trend
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate funding z-score
    funding_z = calculate_funding_zscore(prices, symbol, lookback=30)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if atr_14[i] == 0 or np.isnan(atr_14[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market - use mean reversion
        is_trending = chop[i] < 38.2  # Trend market - use trend following
        
        # === HTF TREND BIAS ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY FILTER ===
        vol_ratio = atr_14[i] / (atr_50[i] + 1e-10)
        extreme_vol = vol_ratio > 2.5
        
        # === FUNDING CONTRARIAN SIGNAL ===
        funding_extreme_long = funding_z[i] > 2.0  # Crowded longs = bearish
        funding_extreme_short = funding_z[i] < -2.0  # Crowded shorts = bullish
        
        # === MEAN REVERSION SIGNALS (Choppy Regime) ===
        # Long: CRSI < 20 + price near/at lower BB
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        price_near_lower_bb = close[i] <= bb_lower[i] * 1.005
        price_near_upper_bb = close[i] >= bb_upper[i] * 0.995
        
        # === TREND FOLLOWING SIGNALS (Trending Regime) ===
        # Long pullback: 12h HMA bullish + RSI(14) 40-55
        rsi_pullback_long = 40.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 60.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if not extreme_vol:
            # --- MEAN REVERSION (Choppy Regime) ---
            if is_choppy:
                # Long: CRSI oversold + price at lower BB
                if crsi_oversold and price_near_lower_bb:
                    new_signal = POSITION_SIZE_BASE
                    if funding_extreme_short:
                        new_signal = POSITION_SIZE_MAX
                
                # Short: CRSI overbought + price at upper BB
                elif crsi_overbought and price_near_upper_bb:
                    new_signal = -POSITION_SIZE_BASE
                    if funding_extreme_long:
                        new_signal = -POSITION_SIZE_MAX
            
            # --- TREND FOLLOWING (Trending Regime) ---
            elif is_trending:
                # Long: 12h/1d HMA bullish + RSI pullback
                if price_above_hma_12h and price_above_hma_1d and rsi_pullback_long:
                    new_signal = POSITION_SIZE_BASE
                    if funding_extreme_short:
                        new_signal = POSITION_SIZE_MAX
                
                # Short: 12h/1d HMA bearish + RSI pullback
                elif price_below_hma_12h and price_below_hma_1d and rsi_pullback_short:
                    new_signal = -POSITION_SIZE_BASE
                    if funding_extreme_long:
                        new_signal = -POSITION_SIZE_MAX
            
            # --- TRANSITION ZONE (38.2 <= CHOP <= 61.8) ---
            else:
                # Use conservative mean reversion with HTF filter
                if crsi_oversold and price_near_lower_bb and price_above_hma_1d:
                    new_signal = POSITION_SIZE_BASE * 0.6
                elif crsi_overbought and price_near_upper_bb and price_below_hma_1d:
                    new_signal = -POSITION_SIZE_BASE * 0.6
        
        # === HOLD POSITION LOGIC ===
        # Keep position if CRSI hasn't reached exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and crsi[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 30.0:
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime switches from mean-revert to strong trend down
        if in_position and position_side > 0:
            if is_trending and price_below_hma_12h and price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if regime switches from mean-revert to strong trend up
        if in_position and position_side < 0:
            if is_trending and price_above_hma_12h and price_above_hma_1d:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals