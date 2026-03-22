#!/usr/bin/env python3
"""
Experiment #320: 1h Primary + 4h/12h HTF — Connors RSI Mean Reversion + HTF Trend Filter

Hypothesis: Connors RSI (CRSI) mean reversion with HTF trend filter works on 1h because:
1. CRSI combines 3 components (RSI(3) + StreakRSI(2) + PercentRank(100)) for superior mean reversion signals
2. 4h HMA(21) provides trend direction without excessive lag
3. 12h ADX filters out extreme chop where mean reversion fails
4. Session filter (8-20 UTC) captures high-liquidity periods only
5. Volume confirmation (>0.7x avg) ensures real moves, not noise
6. Target: 40-80 trades/year on 1h (appropriate for hourly, manageable fee drag)

Why this might beat current best (Sharpe=0.424):
- CRSI has 75% win rate in backtests (vs 55-60% for standard RSI)
- 4h trend filter prevents counter-trend mean reversion (major failure mode)
- Session filter avoids low-liquidity Asian night hours (whipsaw reduction)
- Looser CRSI thresholds (15/85 vs 10/90) ensure adequate trade frequency
- Asymmetric sizing (longs 0.25, shorts 0.20) matches crypto long bias

Key differences from failed 1h strategies (#310, #315, #318):
- Connors RSI instead of standard RSI (better mean reversion signal)
- 4h HMA trend filter (not 1h HMA - reduces whipsaw)
- Session filter 8-20 UTC (avoids dead hours)
- Looser entry thresholds to ensure 40+ trades/year
- Volume filter moderate (0.7x not 1.5x - too strict kills trades)

Position sizing: 0.20 base, 0.25 strong conviction
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_hma_session_vol_4h12h_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of past returns lower than current
    
    Entry: CRSI < 15 (long), CRSI > 85 (short)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
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
        streak_window = streak[max(0, i-streak_period):i+1]
        up_streaks = np.sum(streak_window > 0)
        down_streaks = np.sum(streak_window < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100.0 * up_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    # Component 3: Percent Rank
    percent_rank = np.zeros(n)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100.0
    
    for i in range(rank_period, n):
        window = returns[max(0, i-rank_period):i]
        if len(window) > 0:
            count_lower = np.sum(window < returns[i])
            percent_rank[i] = 100.0 * count_lower / len(window)
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    for i in range(max(rsi_period, streak_period, rank_period), n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    wma_diff = 2.0 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    plus_di = 100.0 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Calculate 12h HTF indicators (regime filter)
    adx_12h_14 = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    adx_12h_14_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    crsi_3_2_100 = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_50 = calculate_hma(close, period=50)
    sma_200 = calculate_sma(close, 200)
    
    # Volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.20
    LONG_STRONG = 0.25
    SHORT_BASE = 0.18
    SHORT_STRONG = 0.22
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(adx_12h_14_aligned[i]):
            continue
        
        if np.isnan(crsi_3_2_100[i]):
            continue
        
        if np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        
        # === 4H TREND DIRECTION (primary filter) ===
        # Bull: price above 4h HMA(21)
        # Bear: price below 4h HMA(21)
        trend_bull = close[i] > hma_4h_21_aligned[i]
        trend_bear = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope (3-bar lookback on aligned array)
        hma_slope_up = hma_4h_21_aligned[i] > hma_4h_21_aligned[i-3] if i >= 3 else False
        hma_slope_down = hma_4h_21_aligned[i] < hma_4h_4h_21_aligned[i-3] if i >= 3 else False
        
        # === 12H ADX REGIME (avoid extreme chop) ===
        # ADX > 25 = trending (favor trend entries)
        # ADX < 20 = choppy (favor mean reversion)
        adx_value = adx_12h_14_aligned[i]
        is_trending_regime = adx_value > 25.0
        is_choppy_regime = adx_value < 20.0
        
        # === 1H LOCAL TREND ===
        hma_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        # CRSI < 15 = oversold (long opportunity)
        # CRSI > 85 = overbought (short opportunity)
        crsi_oversold = crsi_3_2_100[i] < 15.0
        crsi_overbought = crsi_3_2_100[i] > 85.0
        crsi_extreme_oversold = crsi_3_2_100[i] < 10.0
        crsi_extreme_overbought = crsi_3_2_100[i] > 90.0
        
        # CRSI turning (momentum shift)
        crsi_rising = crsi_3_2_100[i] > crsi_3_2_100[i-1] if i > 0 else False
        crsi_falling = crsi_3_2_100[i] < crsi_3_2_100[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        vol_confirmed = vol_ratio > 0.7  # At least 70% of average
        
        # === SESSION FILTER (8-20 UTC) ===
        # Convert open_time to hour
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull trend - asymmetric)
        if trend_bull:
            # CRSI oversold + 4h bull + volume confirmed + in session
            if crsi_oversold and vol_confirmed and in_session:
                if hma_bullish:  # +1 confluence: 1h trend aligns
                    new_signal = LONG_BASE * vol_scale
                elif price_above_sma200:  # +1 confluence: above long-term MA
                    new_signal = LONG_BASE * vol_scale
            
            # Extreme CRSI oversold + 4h bull (stronger signal)
            if crsi_extreme_oversold and trend_bull:
                new_signal = LONG_STRONG * vol_scale
            
            # CRSI turning up from oversold + 4h HMA slope up
            if crsi_rising and crsi_3_2_100[i] < 25.0 and hma_slope_up:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * vol_scale
        
        # SHORT ENTRIES (only in bear trend, reduced size)
        if trend_bear:
            # CRSI overbought + 4h bear + volume confirmed + in session
            if crsi_overbought and vol_confirmed and in_session:
                if hma_bearish:  # +1 confluence: 1h trend aligns
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * vol_scale
                elif not price_above_sma200:  # +1 confluence: below long-term MA
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * vol_scale
            
            # Extreme CRSI overbought + 4h bear (stronger signal)
            if crsi_extreme_overbought and trend_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG * vol_scale
            
            # CRSI turning down from overbought + 4h HMA slope down
            if crsi_falling and crsi_3_2_100[i] > 75.0 and hma_slope_down:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 40+ trades/year on 1h) ===
        # Force trade if no signal for 48 bars (~48 hours = 2 days)
        if bars_since_last_trade > 48 and new_signal == 0.0 and not in_position:
            if crsi_extreme_oversold and trend_bull:
                new_signal = LONG_BASE * 0.7 * vol_scale
            elif crsi_extreme_overbought and trend_bear:
                new_signal = -SHORT_BASE * 0.7 * vol_scale
            elif crsi_oversold and trend_bull and in_session:
                new_signal = LONG_BASE * 0.6 * vol_scale
            elif crsi_overbought and trend_bear and in_session:
                new_signal = -SHORT_BASE * 0.6 * vol_scale
        
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
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI turns overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h trend turns bearish
            if position_side > 0 and trend_bear and close[i] < hma_1h_21[i]:
                trend_reversal = True
            # Short position but 4h trend turns bullish
            if position_side < 0 and trend_bull and close[i] > hma_1h_21[i]:
                trend_reversal = True
        
        if stoploss_triggered or crsi_exit or trend_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.23:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.20:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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