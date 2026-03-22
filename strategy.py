#!/usr/bin/env python3
"""
Experiment #005: 1h Connors RSI + 4h HMA Trend + Choppiness Regime Filter

Hypothesis: 1h primary with 4h/1d HTF trend filter will capture mean-reversion entries
within the major trend direction. Key design:
1. 4h HMA(21) for major trend direction (call ONCE before loop via mtf_data)
2. 1d HMA(50) for regime confirmation (bull/bear market)
3. Connors RSI(3,2,100) for entry timing - proven 75% win rate on reversals
4. Choppiness Index(14) for regime detection - CHOP>55=range(mean revert), CHOP<45=trend
5. Session filter (8-20 UTC) for liquidity, volume > 0.8x avg
6. ATR(14) for stoploss (2.5x) and position sizing
7. Discrete sizing: 0.25 base, 0.30 strong confluence, 0.20 weak

Why this should work:
- Connors RSI catches oversold/overbought reversals within trend (not counter-trend)
- Choppiness filter avoids mean-reversion in strong trends (major failure mode)
- 4h/1d HMA prevents counter-trend trades (2022 crash protection)
- Session filter reduces noise during low-liquidity hours
- 1h TF targets 30-60 trades/year (optimal for fee efficiency)
- Conservative sizing (0.20-0.30) protects from 2022-style crashes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_4h_hma_chop_regime_v1"
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
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) on close - short-term momentum
    2. RSI(2) on streak - consecutive up/down days
    3. PercentRank(100) - where current price ranks in last 100 bars
    
    CRSI < 10 = extremely oversold (long opportunity)
    CRSI > 90 = extremely overbought (short opportunity)
    """
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100 * count_below / rank_period
    
    # Combine components
    crsi = (rsi_close + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP)
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in the window
        atr_sum = 0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_session_filter(open_time):
    """
    Filter for high-liquidity session (8-20 UTC).
    Returns 1 during session, 0 outside.
    """
    # open_time is in milliseconds since epoch
    hours = (open_time // 3600000) % 24
    session_active = (hours >= 8) & (hours < 20)
    return session_active.astype(float)

def calculate_volume_ma(volume, period=20):
    """Calculate moving average of volume."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

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
    
    # Calculate 4h HMA trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d HMA for regime confirmation
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    session = calculate_session_filter(open_time)
    vol_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
    # Track position state for stoploss
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        # === 1D HTF REGIME (Bull/Bear Market) ===
        # Price above 1d HMA(50) = bull market (prefer longs)
        # Price below 1d HMA(50) = bear market (prefer shorts)
        bull_regime = close[i] > hma_1d_50_aligned[i]
        bear_regime = close[i] < hma_1d_50_aligned[i]
        
        # === 4H HTF TREND BIAS ===
        htf_bullish = close[i] > hma_4h_21_aligned[i]
        htf_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        choppy_regime = chop[i] > 55  # range market - mean reversion works
        trending_regime = chop[i] < 45  # trending market - trend follow works
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15  # extreme oversold
        crsi_overbought = crsi[i] > 85  # extreme overbought
        
        # === SESSION FILTER ===
        session_active = session[i] > 0.5
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_ma[i] if not np.isnan(vol_ma[i]) else True
        
        # === POSITION SIZING BASED ON CONFLUENCE ===
        # Count confluence factors
        confluence_long = 0
        confluence_short = 0
        
        if htf_bullish:
            confluence_long += 1
        if htf_bearish:
            confluence_short += 1
        if bull_regime:
            confluence_long += 1
        if bear_regime:
            confluence_short += 1
        if crsi_oversold:
            confluence_long += 1
        if crsi_overbought:
            confluence_short += 1
        if session_active:
            confluence_long += 0.5
            confluence_short += 0.5
        if volume_ok:
            confluence_long += 0.5
            confluence_short += 0.5
        
        # Determine position size based on confluence
        if confluence_long >= 3.5 or confluence_short >= 3.5:
            current_size = STRONG_SIZE
        elif confluence_long >= 2.5 or confluence_short >= 2.5:
            current_size = BASE_SIZE
        else:
            current_size = WEAK_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h bullish + CRSI oversold + (choppy OR bull regime) + session + volume
        if htf_bullish and crsi_oversold:
            if (choppy_regime or bull_regime) and session_active and volume_ok:
                new_signal = current_size
        
        # SHORT ENTRY: 4h bearish + CRSI overbought + (choppy OR bear regime) + session + volume
        elif htf_bearish and crsi_overbought:
            if (choppy_regime or bear_regime) and session_active and volume_ok:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~2 days on 1h), allow weaker entry
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if htf_bullish and crsi_oversold and session_active:
                new_signal = current_size * 0.8
            elif htf_bearish and crsi_overbought and session_active:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === CRSI MEAN REVERSION EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought (mean reversion complete)
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            # Exit short when CRSI becomes oversold (mean reversion complete)
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or crsi_exit:
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