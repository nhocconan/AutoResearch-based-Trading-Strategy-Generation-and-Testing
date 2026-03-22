#!/usr/bin/env python3
"""
Experiment #235: 1h Primary + 4h/1d HTF — Session-Filtered Regime Adaptive

Hypothesis: 1h strategies fail due to excessive trades (>200/yr) causing fee drag.
Solution: Use 4h/1d for TREND DIRECTION, 1h only for ENTRY TIMING with strict filters:
1. 4h HMA(21) slope determines bull/bear regime
2. 1d Choppiness Index filters range vs trend markets
3. 1h Connors RSI for precise entry timing (CRSI < 10 long, > 90 short)
4. Session filter: only trade 8-20 UTC (highest liquidity, lowest slippage)
5. Volume filter: volume > 0.8x 20-bar average
6. Discrete sizing: 0.20 base, 0.30 strong confluence

Key insight from 234 experiments: BTC/ETH 2025+ is bear/range, not trend.
Mean reversion in range markets + trend following in trending markets = adaptive edge.
Position size 0.20-0.25 (smaller for 1h to reduce fee impact).
Target: 40-80 trades/year per symbol (within 1h cost model).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_session_connors_chop_hma_4h1d_v1"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

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
    Long: CRSI < 10 | Short: CRSI > 90
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period+1):i+1]
        positive = np.sum(streak_window > 0)
        if streak_period > 0:
            streak_rsi[i] = (positive / streak_period) * 100
        else:
            streak_rsi[i] = 50
    
    # Percent Rank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        if len(window) > 1:
            rank = np.sum(window[:-1] < window[-1]) / (len(window) - 1)
            percent_rank[i] = rank * 100
        else:
            percent_rank[i] = 50
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    return crsi

def extract_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        # open_time is in milliseconds
        ts_sec = open_time_array[i] / 1000.0
        hours[i] = int((ts_sec % 86400) / 3600)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract hour for session filter
    hours = extract_hour_from_open_time(open_time)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (primary trend regime)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    
    # Calculate 1d HTF indicators (choppiness regime)
    chop_1d = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        14
    )
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    hma_1h_21 = calculate_hma(close, 21)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        if np.isnan(chop_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 4H TREND REGIME ===
        # Bull: slope > 0.2%, Bear: slope < -0.2%
        regime_bull = hma_4h_slope_aligned[i] > 0.20
        regime_bear = hma_4h_slope_aligned[i] < -0.20
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D CHOPPINESS REGIME ===
        # CHOP > 55 = range (mean revert), CHOP < 45 = trend (trend follow)
        is_choppy = chop_1d_aligned[i] > 55.0
        is_trending = chop_1d_aligned[i] < 45.0
        
        # === 1H LOCAL SIGNALS ===
        price_above_1h_hma = close[i] > hma_1h_21[i]
        price_below_1h_hma = close[i] < hma_1h_21[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 12  # Strong mean reversion long
        crsi_overbought = crsi[i] > 88  # Strong mean reversion short
        crsi_mild_oversold = crsi[i] < 25
        crsi_mild_overbought = crsi[i] > 75
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Require session + volume confluence for ALL entries
        if not (in_session and volume_ok):
            signals[i] = 0.0
            continue
        
        # MEAN REVERSION MODE (when choppy on 1d)
        if is_choppy:
            # LONG: CRSI extreme oversold + price below 4h HMA + not in strong bear regime
            if crsi_oversold and price_below_4h_hma and not regime_bear:
                new_signal = STRONG_SIZE
            # LONG: CRSI mild oversold + price below 1h HMA + regime neutral
            elif crsi_mild_oversold and price_below_1h_hma and regime_neutral:
                new_signal = BASE_SIZE
            
            # SHORT: CRSI extreme overbought + price above 4h HMA + not in strong bull regime
            if crsi_overbought and price_above_4h_hma and not regime_bull:
                new_signal = -STRONG_SIZE
            # SHORT: CRSI mild overbought + price above 1h HMA + regime neutral
            elif crsi_mild_overbought and price_above_1h_hma and regime_neutral:
                new_signal = -BASE_SIZE
        
        # TREND FOLLOWING MODE (when trending on 1d)
        if is_trending:
            # LONG: Regime bull + price above 4h HMA + CRSI not overbought
            if regime_bull and price_above_4h_hma and crsi[i] < 70:
                new_signal = BASE_SIZE
            # LONG: Regime bull + price above both HMAs + CRSI bullish (>40)
            elif regime_bull and price_above_4h_hma and price_above_1h_hma and crsi[i] > 40:
                new_signal = STRONG_SIZE
            
            # SHORT: Regime bear + price below 4h HMA + CRSI not oversold
            if regime_bear and price_below_4h_hma and crsi[i] > 30:
                new_signal = -BASE_SIZE
            # SHORT: Regime bear + price below both HMAs + CRSI bearish (<60)
            elif regime_bear and price_below_4h_hma and price_below_1h_hma and crsi[i] < 60:
                new_signal = -STRONG_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure 10+ trades) ===
        # Only force-trade after 60 bars (~2.5 days on 1h) with no position
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 35 and price_above_4h_hma:
                new_signal = BASE_SIZE * 0.5
            elif regime_bear and crsi[i] > 65 and price_below_4h_hma:
                new_signal = -BASE_SIZE * 0.5
            elif is_choppy and crsi_oversold:
                new_signal = BASE_SIZE * 0.4
            elif is_choppy and crsi_overbought:
                new_signal = -BASE_SIZE * 0.4
        
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
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_4h_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_4h_hma:
                regime_reversal = True
        
        # === CRSI REVERSAL EXIT (take profit on mean reversion) ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 70:  # Long profit at CRSI > 70
                crsi_exit = True
            if position_side < 0 and crsi[i] < 30:  # Short profit at CRSI < 30
                crsi_exit = True
        
        if stoploss_triggered or regime_reversal or crsi_exit:
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