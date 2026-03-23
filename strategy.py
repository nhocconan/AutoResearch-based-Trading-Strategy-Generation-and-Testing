#!/usr/bin/env python3
"""
Experiment #078: 30m Primary + 4h/1d HTF — Session-Filtered Pullback Strategy

Hypothesis: 30m timeframe with dual HTF (4h trend + 1d macro) using Connors RSI for 
pullback entries ONLY during London/NY session (8-20 UTC), with volume confirmation 
and choppiness regime filter, will generate 30-60 trades/year with Sharpe > 0.486.

Key innovations:
1) Session filter: Only trade 8-20 UTC (London/NY overlap = highest volume)
2) Dual HTF bias: 4h HMA for intermediate trend, 1d HMA for macro direction
3) Connors RSI pullback: Enter on CRSI < 20 (long) or > 80 (short) WITH HTF trend
4) Choppiness regime: Skip entries when CHOP > 55 (ranging = no trend trades)
5) Volume confirmation: volume > 0.8 * SMA(volume, 20) ensures liquidity
6) Asymmetric sizing: 0.25 for longs, 0.20 for shorts (bear market bias)
7) Strict stoploss: 2.0*ATR trailing, exit on HTF trend reversal

Why this should work for 30m:
- Session filter reduces trades by ~60% (only 12h of 24h)
- HTF trend filter prevents counter-trend trades (major failure mode)
- CRSI pullback entries have 70%+ win rate in trending markets
- Volume filter reduces false breakouts during low liquidity
- Lower position size (0.20-0.25) controls drawdown on lower TF

Position size: 0.20-0.25 (smaller for 30m vs 4h)
Stoploss: 2.0*ATR trailing
Target: 30-60 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_session_hma_regime_4h1d_v1"
timeframe = "30m"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI component
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        streak_window = streak[i-streak_period+1:i+1]
        up_streaks = np.sum(streak_window > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100 if streak_period > 0 else 50
    
    # Percent Rank component
    pr = np.zeros(len(close))
    for i in range(pr_period, len(close)):
        returns = close_s.iloc[i-pr_period+1:i+1].pct_change().dropna()
        if len(returns) > 0:
            current_return = close_s.iloc[i] / close_s.iloc[i-1] - 1
            pr[i] = (np.sum(returns < current_return) / len(returns)) * 100
        else:
            pr[i] = 50
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + pr) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    return crsi

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

def get_utc_hour(open_time_ms):
    """Extract UTC hour from Unix timestamp in milliseconds."""
    return (open_time_ms // (1000 * 3600)) % 24

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
    
    # Calculate 4h HMA for intermediate trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume SMA for confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_LONG = 0.25
    POSITION_SIZE_SHORT = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(vol_sma_20[i]) or atr_14[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === HTF TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Strong trend = both 4h and 1d agree
        strong_bullish = price_above_hma_4h and price_above_hma_1d
        strong_bearish = price_below_hma_4h and price_below_hma_1d
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_trending = chop_value < 50.0  # Below 50 = trending market
        
        # === VOLUME CONFIRMATION ===
        vol_confirms = volume[i] > (0.8 * vol_sma_20[i])
        
        # === CONNORS RSI PULLBACK SIGNALS ===
        crsi_oversold = crsi[i] < 20.0  # Deep pullback
        crsi_overbought = crsi[i] > 80.0  # Deep rally
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours
        if in_session and is_trending and vol_confirms:
            # Long: HTF bullish + CRSI oversold pullback
            if strong_bullish and crsi_oversold:
                new_signal = POSITION_SIZE_LONG
            
            # Short: HTF bearish + CRSI overbought rally
            elif strong_bearish and crsi_overbought:
                new_signal = -POSITION_SIZE_SHORT
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold long if RSI not overbought and HTF still bullish
            if position_side > 0 and rsi_14[i] < 70.0 and price_above_hma_4h:
                new_signal = signals[i-1] if i > 0 else 0.0
            # Hold short if RSI not oversold and HTF still bearish
            elif position_side < 0 and rsi_14[i] > 30.0 and price_below_hma_4h:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON HTF TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h:  # 4h trend broken
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h:  # 4h trend broken
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