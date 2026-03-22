#!/usr/bin/env python3
"""
Experiment #308: 30m Primary + 4h/1d HTF — HMA Trend + Connors RSI + Session Filter

Hypothesis: For 30m timeframe, use 4h/1d for TREND DIRECTION, 30m only for ENTRY TIMING.
This gives HTF trade frequency (20-50 trades/year) with 30m execution precision.

Key components:
1. 4h HMA(21) = major trend direction (long only when price > 4h HMA)
2. 1d Choppiness Index = regime filter (avoid trend entries when CHOP > 55)
3. 30m Connors RSI = entry timing (CRSI < 15 long, CRSI > 85 short)
4. Session filter = only trade 8-20 UTC (reduces trades by ~60%)
5. Volume filter = volume > 0.8x 20-bar average (confirms moves)
6. Stoploss = 2.5 * ATR trailing (signal → 0 when hit)

Why this might work:
- 4h trend filter prevents counter-trend trades (main failure mode)
- Connors RSI (not standard RSI) catches short-term reversals better
- Session filter drastically reduces trade count (critical for 30m)
- Discrete signals (0.0, ±0.20, ±0.30) minimize fee churn
- Asymmetric sizing (longs 0.30, shorts 0.20) matches crypto behavior

Position sizing: 0.20 base, 0.30 strong conviction (longs), 0.20 (shorts)
Target: 40-80 trades/year on 30m (with session filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_connors_chop_session_4h1d_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much more responsive than EMA with less lag.
    """
    n = period
    n2 = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    wma_half = wma(close_s, n2)
    wma_full = wma(close_s, n)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
    return hma.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): 3-period RSI on price
    RSI(streak, 2): 2-period RSI on up/down streak length
    PercentRank(100): percentile rank of today's return over last 100 days
    
    CRSI < 10 = extremely oversold (long)
    CRSI > 90 = extremely overbought (short)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(close, 3)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain_3 = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss_3 = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs_3 = avg_gain_3 / (avg_loss_3 + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs_3))
    
    # RSI(streak, 2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100)
    returns = close_s.pct_change() * 100.0
    percent_rank = pd.Series(index=range(n), dtype=float)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if pd.isna(current):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current).sum()
            percent_rank.iloc[i] = 100.0 * rank / rank_period
    
    percent_rank = percent_rank.fillna(50.0).values
    
    # Combine
    for i in range(n):
        if i >= rank_period:
            crsi[i] = (rsi_3.iloc[i] + rsi_streak.iloc[i] + percent_rank[i]) / 3.0
        else:
            crsi[i] = 50.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_ma + 1e-10)
    return vol_ratio.values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts = pd.to_datetime(open_time, unit='ms', utc=True)
    return ts.dt.hour.values

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
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Calculate 1d HTF indicators (regime filter)
    chop_1d_14 = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    chop_1d_14_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_14)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_30m_21 = calculate_hma(close, 21)
    sma_200 = calculate_sma(close, 200)
    connors_rsi = calculate_connors_rsi(close, 3, 2, 100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
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
        
        if np.isnan(chop_1d_14_aligned[i]):
            continue
        
        if np.isnan(connors_rsi[i]) or np.isnan(hma_30m_21[i]):
            continue
        
        # === 4H MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 4h HMA (favor longs)
        # Bear: price below 4h HMA (allow shorts)
        regime_bull = close[i] > hma_4h_21_aligned[i]
        regime_bear = close[i] < hma_4h_21_aligned[i]
        
        # === 1D CHOPPINESS REGIME ===
        # CHOP > 55 = range market (reduce size, mean revert only)
        # CHOP < 45 = trending market (full size, trend follow)
        is_choppy = chop_1d_14_aligned[i] > 55.0
        is_trending = chop_1d_14_aligned[i] < 45.0
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Reduces trades by ~60%, focuses on high-liquidity hours
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME FILTER ===
        # volume > 0.8x average confirms move
        volume_ok = vol_ratio[i] > 0.8
        
        # === 30M LOCAL TREND ===
        # HMA direction
        hma_bullish = hma_30m_21[i] > hma_30m_21[i-3] if i >= 3 else False
        hma_bearish = hma_30m_21[i] < hma_30m_21[i-3] if i >= 3 else False
        
        # Price relative to HMA
        price_above_hma = close[i] > hma_30m_21[i]
        price_below_hma = close[i] < hma_30m_21[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = extremely oversold (long)
        # CRSI > 85 = extremely overbought (short)
        crsi_oversold = connors_rsi[i] < 15.0
        crsi_overbought = connors_rsi[i] > 85.0
        crsi_neutral = 35.0 < connors_rsi[i] < 65.0
        
        # === ENTRY LOGIC (ASYMMETRIC + REGIME-AWARE + SESSION) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Must be in session for entries (reduces trade count)
        if in_session and volume_ok:
            # LONG ENTRIES (favored in bull regime)
            if regime_bull:
                # Connors RSI oversold + trending market + 4h bull
                if is_trending and crsi_oversold and hma_bullish:
                    new_signal = LONG_STRONG
                
                # Connors RSI oversold + choppy market (mean revert)
                elif is_choppy and crsi_oversold and price_below_hma:
                    new_signal = LONG_BASE
                
                # Moderate CRSI + bull regime + above SMA200
                elif connors_rsi[i] < 25.0 and regime_bull and price_above_sma200:
                    new_signal = LONG_BASE
                
                # HMA bullish + CRSI rising from oversold
                elif hma_bullish and connors_rsi[i] < 30.0 and connors_rsi[i] > connors_rsi[i-1]:
                    new_signal = LONG_BASE
            
            # SHORT ENTRIES (only in bear regime, reduced size)
            if regime_bear:
                # Connors RSI overbought + trending market
                if is_trending and crsi_overbought and hma_bearish:
                    if new_signal == 0.0:
                        new_signal = -SHORT_STRONG
                
                # Connors RSI overbought + choppy market (mean revert)
                elif is_choppy and crsi_overbought and price_above_hma:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE
                
                # Moderate CRSI + bear regime
                elif connors_rsi[i] > 75.0 and regime_bear:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE
                
                # HMA bearish + CRSI falling from overbought
                elif hma_bearish and connors_rsi[i] > 70.0 and connors_rsi[i] < connors_rsi[i-1]:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        # Force trade if no signal for 50 bars (~25 hours on 30m)
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if regime_bull and connors_rsi[i] < 35.0:
                new_signal = LONG_BASE * 0.6
            elif regime_bear and connors_rsi[i] > 65.0:
                new_signal = -SHORT_BASE * 0.6
        
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
        
        # === CONNORS RSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI turns overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h regime turns bearish
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 4h regime turns bullish
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.15:
                new_signal = 0.0
            elif new_signal > 0.27:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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