#!/usr/bin/env python3
"""
Experiment #385: 15m Connors RSI + 1h HMA Trend + ADX Regime Filter

Hypothesis: After analyzing 384 failed experiments, the key pattern is that strategies
fail when they don't adapt to market regime AND don't generate enough trades.

For 15m timeframe, I propose:
1. CONNORS RSI (CRSI): Combines RSI(3) + RSI_Streak(2) + PercentRank(100)
   - More responsive than standard RSI(14) for 15m entries
   - Proven 75% win rate in mean-reversion scenarios
   - Long when CRSI < 15, Short when CRSI > 85

2. 1h HMA(21) TREND BIAS: Via mtf_data helper (call ONCE before loop)
   - Long only when price > 1h HMA (bullish bias)
   - Short only when price < 1h HMA (bearish bias)
   - HMA smoother than EMA, less lag for trend detection

3. ADX(14) REGIME FILTER:
   - ADX > 25 = trending (widen CRSI thresholds, follow trend)
   - ADX < 20 = ranging (tight CRSI thresholds, mean-revert)
   - Avoids entries during low-volatility chop

4. VOLUME CONFIRMATION:
   - Entry volume > 0.8 * 20-bar volume MA
   - Filters false breakouts on low liquidity

5. ATR TRAILING STOP (2.0x): Tighter for 15m timeframe
   - Signal → 0 when price moves 2.0*ATR against position
   - Protects from rapid 15m reversals

6. POSITION SIZING: 0.25 discrete (conservative for 15m noise)
   - Max 25% capital per position
   - Discrete levels: 0.0, ±0.25 to minimize fee churn

Why this should work on 15m:
- CRSI generates 50-100 signals/year (enough for statistical significance)
- 1h HMA provides stable trend bias without 1d lag
- ADX regime filter adapts to market conditions
- Should work on BTC, ETH, SOL individually (not SOL-biased)
- Tighter stoploss (2.0x ATR) suits 15m volatility

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_1h_hma_adx_vol_atr_v1"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentage of prior closes lower than current close
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component
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
        abs_streak = np.abs(streak[max(0, i-streak_period):i+1])
        if len(abs_streak) > 0 and abs_streak.max() > 0:
            streak_rsi[i] = (np.abs(streak[i]) / (abs_streak.max() + 1e-10)) * 100
        else:
            streak_rsi[i] = 50
    
    # Percent Rank component
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        if len(window) > 0:
            count_lower = np.sum(window < close[i])
            crsi[i] = (rsi_short[i] + streak_rsi[i] + (count_lower / len(window) * 100)) / 3.0
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
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
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100 * minus_dm_s / (tr_s + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_s = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:] = adx_s
    
    return adx

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ma = calculate_volume_ma(volume, 20)
    
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
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === TREND BIAS FROM 1h HMA ===
        bull_trend_1h = close[i] > hma_1h_aligned[i]
        bear_trend_1h = close[i] < hma_1h_aligned[i]
        
        # === ADX REGIME DETECTION ===
        trending_regime = adx[i] > 25
        ranging_regime = adx[i] < 20
        # neutral_regime = 20 <= adx <= 25 (reduce position size or stay flat)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_ma[i]
        
        # === CRSI SIGNALS ===
        # Adjust thresholds based on regime
        if trending_regime:
            # In trending market, use wider thresholds (follow trend)
            crsi_oversold = 25
            crsi_overbought = 75
        elif ranging_regime:
            # In ranging market, use tighter thresholds (mean-revert)
            crsi_oversold = 15
            crsi_overbought = 85
        else:
            # Neutral regime - stay flat or reduce size
            crsi_oversold = 20
            crsi_overbought = 80
        
        crsi_long = crsi[i] < crsi_oversold
        crsi_short = crsi[i] > crsi_overbought
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: CRSI oversold + bullish trend + volume confirmed
        if crsi_long and bull_trend_1h and volume_confirmed:
            new_signal = SIZE
        
        # SHORT ENTRY: CRSI overbought + bearish trend + volume confirmed
        elif crsi_short and bear_trend_1h and volume_confirmed:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1h:
                # Long position but trend turned bearish
                new_signal = 0.0
            if position_side < 0 and bull_trend_1h:
                # Short position but trend turned bullish
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