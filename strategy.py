#!/usr/bin/env python3
"""
Experiment #082: 12h Primary + 1d/1w HTF — Dual Regime with Connors RSI + Choppiness

Hypothesis: 12h timeframe with regime-switching reduces whipsaws vs 4h. 
Choppiness Index detects choppy (mean revert) vs trending (trend follow) regimes.
Connors RSI for mean reversion has 75% win rate in research.
1d/1w HMA provides macro trend bias to avoid counter-trend trades in bear markets.
Funding rate z-score adds contrarian edge for BTC/ETH.

Key innovations:
1) Choppiness Index(14) regime: CHOP>61.8=range(mean revert), CHOP<38.2=trend(trend follow)
2) Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3) 1d HMA(21) + 1w HMA(21) for multi-level trend bias
4) Funding rate z-score(30) contrarian filter
5) ATR(14) volatility filter — skip extreme vol (ratio>2.0)
6) Discrete sizing: 0.25 base, 0.30 max with funding boost
7) 2.5*ATR trailing stoploss

Why 12h should work:
- Lower frequency = 20-50 trades/year = less fee drag (1-2.5%)
- Regime detection avoids trend strategies in choppy markets
- Connors RSI extremes catch reversals better than standard RSI(14)
- 1w HMA prevents counter-trend trades during major bear markets (2022, 2025)

Position size: 0.25 base, 0.30 max with funding boost
Stoploss: 2.5*ATR trailing
Target: 25-45 trades/year, Sharpe > 0.5, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d1w_funding_v1"
timeframe = "12h"
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
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    atr_vals = calculate_atr(high, low, close, period=period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    # Choppiness formula
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of daily returns over lookback
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streaks
    returns = close_s.pct_change()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Component 3: PercentRank of returns
    returns_np = returns.fillna(0.0).values
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = returns_np[i-rank_period+1:i+1]
        current = returns_np[i]
        rank = np.sum(window < current) / len(window) * 100.0
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d HMA slope
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] > 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 1w HMA slope
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]) and hma_1w_aligned[i-1] > 0:
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1] * 100
        else:
            hma_1w_slope[i] = 0.0
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, period=14)
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or np.isnan(choppiness[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            continue
        if atr_14[i] == 0 or np.isnan(atr_14[i]):
            continue
        
        # === HTF TREND BIAS (1d and 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        hma_1d_slope_positive = hma_1d_slope[i] > 0.05  # 0.05% threshold
        hma_1d_slope_negative = hma_1d_slope[i] < -0.05
        hma_1w_slope_positive = hma_1w_slope[i] > 0.02  # 0.02% threshold
        hma_1w_slope_negative = hma_1w_slope[i] < -0.02
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = choppiness[i] > 61.8  # Range/mean reversion
        trending_regime = choppiness[i] < 38.2  # Trend following
        neutral_regime = not choppy_regime and not trending_regime
        
        # === VOLATILITY FILTER ===
        vol_ratio = atr_14[i] / (atr_50[i] + 1e-10)
        extreme_vol = vol_ratio > 2.0
        
        # === FUNDING CONTRARIAN SIGNAL ===
        funding_extreme_long = funding_z[i] > 2.0  # Crowded longs = bearish
        funding_extreme_short = funding_z[i] < -2.0  # Crowded shorts = bullish
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Strong mean reversion long
        crsi_overbought = crsi[i] > 85.0  # Strong mean reversion short
        crsi_moderate_oversold = crsi[i] < 25.0
        crsi_moderate_overbought = crsi[i] > 75.0
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if not extreme_vol:
            # --- MEAN REVERSION (Choppy Regime) ---
            if choppy_regime:
                # Long: CRSI oversold + price above 1w HMA (bullish macro)
                if crsi_oversold and price_above_hma_1w:
                    new_signal = POSITION_SIZE_BASE
                    if funding_extreme_short:
                        new_signal = POSITION_SIZE_MAX
                
                # Short: CRSI overbought + price below 1w HMA (bearish macro)
                elif crsi_overbought and price_below_hma_1w:
                    new_signal = -POSITION_SIZE_BASE
                    if funding_extreme_long:
                        new_signal = -POSITION_SIZE_MAX
            
            # --- TREND FOLLOWING (Trending Regime) ---
            elif trending_regime:
                # Long: 1d HMA bullish + 1w HMA bullish + EMA bullish + CRSI moderate oversold
                if price_above_hma_1d and hma_1d_slope_positive and price_above_hma_1w and ema_bullish:
                    if crsi_moderate_oversold:
                        new_signal = POSITION_SIZE_BASE
                        if funding_extreme_short:
                            new_signal = POSITION_SIZE_MAX
                
                # Short: 1d HMA bearish + 1w HMA bearish + EMA bearish + CRSI moderate overbought
                elif price_below_hma_1d and hma_1d_slope_negative and price_below_hma_1w and ema_bearish:
                    if crsi_moderate_overbought:
                        new_signal = -POSITION_SIZE_BASE
                        if funding_extreme_long:
                            new_signal = -POSITION_SIZE_MAX
            
            # --- NEUTRAL REGIME (relaxed conditions) ---
            else:
                # Long: CRSI very oversold + price above 1w HMA
                if crsi_oversold and price_above_hma_1w:
                    new_signal = POSITION_SIZE_BASE
                
                # Short: CRSI very overbought + price below 1w HMA
                elif crsi_overbought and price_below_hma_1w:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            if position_side > 0 and crsi[i] < 75.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 25.0:
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
            if price_below_hma_1w and hma_1w_slope_negative:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1w and hma_1w_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
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