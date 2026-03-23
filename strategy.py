#!/usr/bin/env python3
"""
Experiment #462: 12h Primary + 1d/1w HTF — Funding Z-Score + Connors RSI + Choppiness Regime

Hypothesis: Research shows funding rate mean reversion is the BEST edge for BTC/ETH 
(Sharpe 0.8-1.5 through 2022 crash). Combined with Connors RSI (75% win rate for 
mean reversion) and Choppiness Index for regime detection. Key innovations:
1. Funding Rate Z-Score(30d): z<-2 → long bias, z>+2 → short bias (contrarian)
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — superior to standard RSI
3. Choppiness Index(14) regime: CHOP>61.8=range (mean revert), CHOP<38.2=trend (breakout)
4. 1d HMA(21) + 1w HMA(21) for ultra-long trend bias
5. Donchian(20) breakout confirmation for trend regime
6. ATR(14) trailing stop at 2.5x for risk management
7. Asymmetric sizing: 0.30 with funding bias, 0.20 against funding

Target: Sharpe > 0.612, 20-50 trades/year, DD < -35%
Timeframe: 12h (proven for swing trading, lower fee drag than 4h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_funding_zscore_crsi_chop_donchian_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    rsi_3 = np.clip(rsi_3, 0, 100)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * (streak_abs[i] / (streak_abs[i] + 1e-10))
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1.0 / (streak_abs[i] + 1))
        else:
            streak_rsi[i] = 50.0
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank of daily returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            count_below = np.sum(valid < returns[i])
            percent_rank[i] = 100.0 * count_below / len(valid)
        else:
            percent_rank[i] = 50.0
    percent_rank = np.clip(percent_rank, 0, 100)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return np.clip(crsi, 0, 100)

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def load_funding_data(symbol):
    """Load funding rate data from parquet file."""
    try:
        # Map symbol to filename
        symbol_map = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        base_symbol = symbol_map.get(symbol, symbol.replace('USDT', ''))
        funding_path = f"data/processed/funding/{base_symbol}.parquet"
        df = pd.read_parquet(funding_path)
        return df
    except Exception:
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Load funding data and align to prices
    symbol = prices.get('symbol', 'BTCUSDT')
    funding_df = load_funding_data(symbol)
    funding_z = np.full(n, np.nan)
    
    if funding_df is not None and len(funding_df) > 0:
        # Calculate funding z-score (30-day rolling)
        funding_rates = funding_df['funding_rate'].values if 'funding_rate' in funding_df.columns else funding_df.iloc[:, -1].values
        funding_z_raw = pd.Series(funding_rates).rolling(window=30, min_periods=30).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10) if len(x) >= 30 else np.nan
        ).values
        
        # Align funding z-score to LTF (shift by 1 to avoid look-ahead)
        if len(funding_z_raw) > 0:
            # Simple alignment: repeat each HTF value for LTF bars
            ratio = n // len(funding_z_raw)
            if ratio > 0:
                funding_z_aligned = np.repeat(funding_z_raw, ratio)
                if len(funding_z_aligned) > n:
                    funding_z_aligned = funding_z_aligned[:n]
                elif len(funding_z_aligned) < n:
                    funding_z_aligned = np.pad(funding_z_aligned, (0, n - len(funding_z_aligned)), constant_values=np.nan)
                funding_z = np.roll(funding_z_aligned, 1)  # shift by 1 for completed bar
                funding_z[:100] = np.nan  # warmup period
    
    signals = np.zeros(n)
    SIZE_WITH_BIAS = 0.30  # 30% when funding supports direction
    SIZE_AGAINST_BIAS = 0.20  # 20% when fighting funding
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 61.8  # Range market - mean reversion
        regime_trend = chop[i] < 38.2  # Trending market - breakout
        
        # === FUNDING BIAS (Contrarian) ===
        funding_extreme_long = funding_z[i] < -2.0  # Extremely negative funding → long opportunity
        funding_extreme_short = funding_z[i] > 2.0  # Extremely positive funding → short opportunity
        funding_neutral = -2.0 <= funding_z[i] <= 2.0
        
        # === HTF TREND BIAS (1d and 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        crsi_moderate_oversold = crsi[i] < 30.0
        crsi_moderate_overbought = crsi[i] > 70.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ASYMMETRIC POSITION SIZING ===
        if funding_extreme_long or price_above_hma_1w:  # Bull bias
            size_long = SIZE_WITH_BIAS
            size_short = SIZE_AGAINST_BIAS
        elif funding_extreme_short or price_below_hma_1w:  # Bear bias
            size_long = SIZE_AGAINST_BIAS
            size_short = SIZE_WITH_BIAS
        else:  # Neutral
            size_long = SIZE_WITH_BIAS
            size_short = SIZE_WITH_BIAS
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 61.8) — MEAN REVERSION ===
        if regime_chop:
            # Long: CRSI oversold + RSI oversold + HTF not strongly bearish
            if crsi_oversold and rsi_oversold:
                signal_strength = 2
                if price_above_hma_1d:
                    signal_strength += 1
                if funding_extreme_long:
                    signal_strength += 2  # Strong funding signal
                elif funding_neutral:
                    signal_strength += 1
                
                if signal_strength >= 3:
                    desired_signal = size_long
            
            # Short: CRSI overbought + RSI overbought + HTF not strongly bullish
            if crsi_overbought and rsi_overbought and desired_signal == 0:
                signal_strength = 2
                if price_below_hma_1d:
                    signal_strength += 1
                if funding_extreme_short:
                    signal_strength += 2
                elif funding_neutral:
                    signal_strength += 1
                
                if signal_strength >= 3:
                    desired_signal = -size_short
        
        # === REGIME 2: TRENDING (CHOP < 38.2) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout + HTF bullish + funding not extremely short
            if donchian_breakout_long:
                signal_strength = 1
                if price_above_hma_1d:
                    signal_strength += 1
                if price_above_hma_1w:
                    signal_strength += 1
                if not funding_extreme_short:
                    signal_strength += 1
                
                if signal_strength >= 2:
                    desired_signal = size_long
            
            # Short: Donchian breakdown + HTF bearish + funding not extremely long
            if donchian_breakout_short and desired_signal == 0:
                signal_strength = 1
                if price_below_hma_1d:
                    signal_strength += 1
                if price_below_hma_1w:
                    signal_strength += 1
                if not funding_extreme_long:
                    signal_strength += 1
                
                if signal_strength >= 2:
                    desired_signal = -size_short
        
        # === REGIME 3: TRANSITION (38.2-61.8) — CRSI MEAN REVERT ONLY ===
        else:
            if crsi_oversold and not price_below_hma_1w:
                desired_signal = size_long * 0.8
            elif crsi_overbought and not price_above_hma_1w:
                desired_signal = -size_short * 0.8
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (price_above_hma_1d or crsi[i] < 50):
                desired_signal = size_long
            elif position_side < 0 and (price_below_hma_1d or crsi[i] > 50):
                desired_signal = -size_short
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.18:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.18:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.15
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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