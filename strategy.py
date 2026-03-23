#!/usr/bin/env python3
"""
Experiment #183: 1d Primary + 1w HTF — Regime-Adaptive (Chop + Connors RSI + Donchian)

Hypothesis: Previous 1d strategies failed due to overly restrictive entry conditions (0 trades).
This uses a regime-adaptive approach: mean reversion (Connors RSI) in choppy markets,
trend following (Donchian breakout) in trending markets. Choppiness Index detects regime.
1w HMA provides ultra-long-term bias. Key insight: simpler OR logic for entries ensures
30-50 trades/year while regime filter prevents wrong-direction trades.

REGIME LOGIC:
- CHOP(14) > 61.8: Range market → use Connors RSI mean reversion
- CHOP(14) < 38.2: Trend market → use Donchian breakout
- Between 38.2-61.8: Neutral → reduce position size

KEY IMPROVEMENTS:
1. Choppiness Index regime detection (proven for ETH/BTC bear markets)
2. Connors RSI for mean reversion: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. Donchian(20) breakout for trend following
4. 1w HMA for macro bias (only long when price>1w HMA, only short when price<1w HMA)
5. ATR trailing stop at 2.5x for risk management
6. Discrete position sizing: 0.0, ±0.20, ±0.30 to minimize fee churn
7. Looser entry thresholds to ensure 30-50 trades/year

TARGET: 30-50 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_crsi_donchian_chop_1w_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            chop[i] = 50.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / range_val) / np.log10(period)
        chop[i] = np.clip(chop[i], 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
        
        # Convert streak to RSI-like value
        if streak[i] > 0:
            streak_rsi[i] = min(100.0, streak[i] * 50.0 / streak_period)
        elif streak[i] < 0:
            streak_rsi[i] = max(0.0, 100.0 + streak[i] * 50.0 / streak_period)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and returns[-1] != 0:
            rank = np.sum(returns[:-1] < returns[-1])
            percent_rank[i] = 100.0 * rank / (len(returns) - 1)
        else:
            percent_rank[i] = 50.0
    
    # CRSI
    crsi = (rsi + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (upper/lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = rsi.fillna(50.0).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 1w HMA for ultra-long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop_14[i] > 61.8  # Range market
        is_trending = chop_14[i] < 38.2  # Trend market
        is_neutral = not is_choppy and not is_trending
        
        # === HTF MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        new_signal = 0.0
        
        # --- MEAN REVERSION (Choppy Market) ---
        if is_choppy:
            # Long: CRSI < 15 (oversold) + price above 1w HMA (bullish bias)
            if crsi[i] < 15.0 and price_above_hma_1w:
                new_signal = POSITION_SIZE_HALF
            
            # Short: CRSI > 85 (overbought) + price below 1w HMA (bearish bias)
            if crsi[i] > 85.0 and price_below_hma_1w:
                new_signal = -POSITION_SIZE_HALF
        
        # --- TREND FOLLOWING (Trending Market) ---
        if is_trending:
            # Long: Price breaks Donchian upper + price above 1w HMA
            if close[i] > donchian_upper[i-1] and price_above_hma_1w:
                new_signal = POSITION_SIZE_FULL
            
            # Short: Price breaks Donchian lower + price below 1w HMA
            if close[i] < donchian_lower[i-1] and price_below_hma_1w:
                new_signal = -POSITION_SIZE_FULL
        
        # --- NEUTRAL REGIME (reduced size, RSI extremes) ---
        if is_neutral:
            # Long: RSI < 30 + price above 1w HMA
            if rsi_14[i] < 30.0 and price_above_hma_1w:
                new_signal = POSITION_SIZE_HALF
            
            # Short: RSI > 70 + price below 1w HMA
            if rsi_14[i] > 70.0 and price_below_hma_1w:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1w HMA
                if price_above_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1w HMA
                if price_below_hma_1w:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1w HMA (trend changed)
        if in_position and position_side > 0 and price_below_hma_1w:
            new_signal = 0.0
        
        # Exit short if price crosses above 1w HMA (trend changed)
        if in_position and position_side < 0 and price_above_hma_1w:
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
                # Position flip
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