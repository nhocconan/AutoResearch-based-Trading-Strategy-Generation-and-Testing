#!/usr/bin/env python3
"""
Experiment #250: 1h Primary + 4h/12h HTF — Session-Filtered Regime Strategy

Hypothesis: After 249 experiments, the key insight is that 1h strategies fail due to
too many trades (fee drag) OR too few trades (0 Sharpe). The solution is:
1. Use 4h/12h HTF for SIGNAL DIRECTION (proven in #246 12h KAMA success)
2. Use 1h only for ENTRY TIMING within HTF trend
3. Session filter (8-20 UTC) reduces trades by ~60% while keeping quality entries
4. Connors RSI for mean reversion entries in bear/range markets (2025+)
5. Volume confirmation to avoid low-liquidity false signals

This combines:
- 4h HMA(21) for primary trend (proven in baseline mtf_hma_rsi_zscore_v1)
- 12h Choppiness Index for regime detection (from #246 success)
- 1h Connors RSI(3,2,100) for entry timing (75% win rate in literature)
- Session filter 8-20 UTC (reduces fee drag on lower TF)
- Volume > 0.8x 20-bar average (avoids low-liquidity traps)
- ATR(14) 2.5x trailing stoploss

Position sizing: 0.20 base, 0.30 strong (conservative for 1h)
Target: 40-80 trades/year (within 1h cost model of 2-4% fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_session_connors_chop_hma_4h12h_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi.fillna(50).values
    
    # Percent Rank component
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def is_session_active(open_time, start_hour=8, end_hour=20):
    """Check if bar is within trading session (UTC)."""
    # open_time is in milliseconds
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators (primary trend)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_48 = calculate_hma(df_4h['close'].values, 48)
    
    # Calculate 12h HTF indicators (regime)
    chop_12h_14 = calculate_choppiness_index(
        df_12h['high'].values, 
        df_12h['low'].values, 
        df_12h['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    chop_12h_14_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -40
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(chop_12h_14_aligned[i]) or np.isnan(crsi[i]):
            continue
        
        # === SESSION FILTER (CRITICAL for 1h trade frequency) ===
        # Only trade 8-20 UTC (reduces ~60% of bars, keeps quality entries)
        in_session = is_session_active(open_time[i], 8, 20)
        
        # === VOLUME FILTER ===
        # Volume must be > 0.8x 20-bar average
        volume_ok = vol_ratio[i] > 0.8
        
        # === 4H TREND REGIME (primary direction filter) ===
        # Bull: price > 4h HMA(21) > HMA(48)
        # Bear: price < 4h HMA(21) < HMA(48)
        trend_4h_bull = (close[i] > hma_4h_21_aligned[i]) and (hma_4h_21_aligned[i] > hma_4h_48_aligned[i])
        trend_4h_bear = (close[i] < hma_4h_21_aligned[i]) and (hma_4h_21_aligned[i] < hma_4h_48_aligned[i])
        trend_4h_neutral = not trend_4h_bull and not trend_4h_bear
        
        # === 12H CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries via CRSI)
        # CHOP < 45 = trend market (follow 4h trend)
        is_choppy = chop_12h_14_aligned[i] > 55.0
        is_trending = chop_12h_14_aligned[i] < 45.0
        
        # === ENTRY LOGIC (3+ confluence required) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # CONFLUENCE 1: Session active
        # CONFLUENCE 2: Volume OK
        # CONFLUENCE 3: HTF trend aligned
        # CONFLUENCE 4: CRSI extreme
        
        if in_session and volume_ok:
            # TREND FOLLOWING MODE (when 12h trending + 4h trend aligned)
            if is_trending:
                # LONG: 4h bull trend + CRSI pullback (< 35) + volume
                if trend_4h_bull and crsi[i] < 35:
                    new_signal = STRONG_SIZE
                # LONG: 4h bull trend + CRSI moderate (< 45) + strong volume
                elif trend_4h_bull and crsi[i] < 45 and vol_ratio[i] > 1.2:
                    new_signal = BASE_SIZE
                
                # SHORT: 4h bear trend + CRSI rally (> 65) + volume
                if trend_4h_bear and crsi[i] > 65:
                    new_signal = -STRONG_SIZE
                # SHORT: 4h bear trend + CRSI moderate (> 55) + strong volume
                elif trend_4h_bear and crsi[i] > 55 and vol_ratio[i] > 1.2:
                    new_signal = -BASE_SIZE
            
            # MEAN REVERSION MODE (when 12h choppy + CRSI extremes)
            if is_choppy:
                # LONG: Choppy + CRSI very oversold (< 15) + not in strong bear
                if crsi[i] < 15 and not trend_4h_bear:
                    new_signal = BASE_SIZE
                # LONG: Choppy + CRSI oversold (< 25) + 4h neutral
                elif crsi[i] < 25 and trend_4h_neutral:
                    if new_signal == 0.0:
                        new_signal = BASE_SIZE * 0.8
                
                # SHORT: Choppy + CRSI very overbought (> 85) + not in strong bull
                if crsi[i] > 85 and not trend_4h_bull:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE
                # SHORT: Choppy + CRSI overbought (> 75) + 4h neutral
                elif crsi[i] > 75 and trend_4h_neutral:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 40 bars (~40 hours on 1h) but only in session
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position and in_session:
            if trend_4h_bull and crsi[i] < 40:
                new_signal = BASE_SIZE * 0.6
            elif trend_4h_bear and crsi[i] > 60:
                new_signal = -BASE_SIZE * 0.6
            elif is_choppy and crsi[i] < 20:
                new_signal = BASE_SIZE * 0.5
            elif is_choppy and crsi[i] > 80:
                new_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h trend turns strongly bearish
            if position_side > 0 and trend_4h_bear:
                regime_reversal = True
            # Short position but 4h trend turns strongly bullish
            if position_side < 0 and trend_4h_bull:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals