#!/usr/bin/env python3
"""
Experiment #511: 15m Multi-Timeframe Connors RSI with 4h HMA Bias + 1h ADX Regime

Hypothesis: After analyzing 510 failed experiments, the key insight is that 15m timeframe
requires STRONGER higher-timeframe filters to avoid noise whipsaws. This strategy combines:

1. CONNORS RSI (CRSI) - Proven mean-reversion indicator with 75% win rate in literature
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Extreme readings (<10 or >90) signal oversold/overbought conditions

2. 4h HMA(21) TREND BIAS via mtf_data helper:
   - Only long when price > 4h HMA (bullish bias)
   - Only short when price < 4h HMA (bearish bias)
   - HMA smoother than EMA, less lag than SMA

3. 1h ADX(14) REGIME FILTER via mtf_data helper:
   - ADX < 25 = ranging market (enable mean-reversion CRSI signals)
   - ADX >= 25 = trending market (disable mean-reversion, use momentum)
   - Critical for avoiding CRSI whipsaws in strong trends

4. VOLUME CONFIRMATION:
   - Volume must be > 0.8 * 20-bar volume MA (avoid low-liquidity traps)
   - Prevents entries during dead zones

5. ATR(14) TRAILING STOP at 2.5x:
   - Tighter stop for 15m timeframe volatility
   - Signal → 0 when price moves 2.5*ATR against position

6. POSITION SIZING: 0.25 discrete (conservative for 15m noise)
   - Discrete levels minimize fee churn
   - Lower than daily strategies due to higher frequency

Why this should work on 15m:
- Connors RSI is specifically designed for short-term mean reversion
- 4h HMA provides strong trend bias without 15m noise
- 1h ADX filters out trending periods where mean-reversion fails
- Volume filter avoids low-liquidity false signals
- Should generate 50-100 trades/year per symbol (enough for Sharpe)
- Asymmetric logic (different entry for bull/bear) matches BTC/ETH behavior

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h and 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_connors_rsi_4h_hma_1h_adx_volume_atr_v1"
timeframe = "15m"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component of Connors RSI.
    Streak = number of consecutive up/down days
    RSI_Streak = RSI of the streak values
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    delta = streak_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + rs))
    return rsi_streak.values

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of Connors RSI.
    PercentRank = percentage of past period returns less than current return
    """
    n = len(close)
    percent_rank = np.full(n, np.nan)
    
    # Calculate returns
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 0 else 0
    
    for i in range(period, n):
        current_return = returns[i]
        past_returns = returns[i-period+1:i]
        
        count_lower = np.sum(past_returns < current_return)
        percent_rank[i] = 100 * count_lower / period
    
    return percent_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_close = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    volume_s = pd.Series(volume)
    return volume_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_1h = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    volume_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(volume_ma[i]) or volume_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # === 1h ADX REGIME FILTER ===
        adx_value = adx_1h_aligned[i]
        ranging_market = adx_value < 25
        trending_market = adx_value >= 25
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * volume_ma[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 10
        crsi_overbought = crsi[i] > 90
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG entries (only in bull regime + ranging market + volume confirmed)
        if bull_regime and ranging_market and volume_confirmed:
            if crsi_oversold:
                new_signal = SIZE
        
        # SHORT entries (only in bear regime + ranging market + volume confirmed)
        if bear_regime and ranging_market and volume_confirmed:
            if crsi_overbought:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals