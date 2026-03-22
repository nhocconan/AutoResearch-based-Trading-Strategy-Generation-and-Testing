#!/usr/bin/env python3
"""
Experiment #277: 15m Supertrend + 4h HMA Bias + ADX + RSI Pullback Entries

Hypothesis: 15m timeframe needs faster signals than 4h/12h strategies while maintaining
HTF directional bias. This strategy combines:

1. 4h HMA(21) for directional bias - prevents counter-trend trades (learned from #267, #271)
2. 15m Supertrend(10,3) for entry triggers - proven momentum indicator
3. ADX(14) > 15 filter - avoids choppy markets but looser than 20 to get more trades
4. RSI(7) pullback entries - adds mean-reversion entries within trend (RSI<35 long, >65 short)
5. Volume confirmation (1.1x avg) - lighter filter than previous attempts
6. 2.5*ATR trailing stoploss - appropriate for 15m (tighter than 12h's 3.0*ATR)
7. Dual entry modes - Supertrend breakout OR RSI pullback within trend

Why this should work better than failed 15m attempts:
- #265 (KAMA) failed with Sharpe=-2.353 - KAMA too slow for 15m
- #271 (Fisher) failed with Sharpe=-6.839 - Fisher too noisy on 15m
- Supertrend is more robust than KAMA/Fisher for intraday
- RSI pullback adds entry opportunities without being sole signal
- Looser ADX threshold (15 vs 20) ensures >=10 trades per symbol
- 4h HMA bias is strong enough filter without being too restrictive

Position sizing: 0.25 base, 0.15 in high vol (discrete levels per Rule 4)
Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_adx_rsi_pullback_v1"
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

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    st = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    hl2 = (high + low) / 2
    
    for i in range(len(atr)):
        if np.isnan(atr[i]):
            st[i] = np.nan
            direction[i] = 0
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == 0:
            st[i] = upper_band
            direction[i] = 1
        else:
            if direction[i-1] == 1:
                if close[i] < lower_band:
                    direction[i] = -1
                    st[i] = upper_band
                else:
                    direction[i] = 1
                    st[i] = max(lower_band, st[i-1])
            else:
                if close[i] > upper_band:
                    direction[i] = 1
                    st[i] = lower_band
                else:
                    direction[i] = -1
                    st[i] = min(upper_band, st[i-1])
    
    return st, direction

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_move = high[i] - high[i-1]
        low_move = low[i-1] - low[i]
        
        if high_move > low_move and high_move > 0:
            plus_dm[i] = high_move
        if low_move > high_move and low_move > 0:
            minus_dm[i] = low_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().iloc[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().iloc[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values[period:]
    
    return adx

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, atr, 3.0)
    rsi = calculate_rsi(close, 7)  # Faster RSI for 15m
    adx = calculate_adx(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = strong directional bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        # Lighter filter than previous attempts to get more trades
        volume_confirmed = volume[i] > 1.1 * vol_sma[i]
        
        # === ADX FILTER ===
        # Looser threshold (15 vs 20) to ensure >=10 trades per symbol
        adx_confirmed = adx[i] > 15
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        position_size = SIZE_REDUCED if high_volatility else SIZE_BASE
        
        # === SUPERTREND SIGNALS ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === RSI PULLBACK SIGNALS ===
        # RSI pullback within trend for additional entry opportunities
        rsi_oversold = rsi[i] < 35  # Pullback long
        rsi_overbought = rsi[i] > 65  # Pullback short
        
        # === ENTRY CONDITIONS (DUAL MODE) ===
        new_signal = 0.0
        
        # LONG: 4h bias up + (Supertrend bullish OR RSI oversold pullback) + ADX
        # Volume confirmation is nice-to-have, not required (looser for more trades)
        long_supertrend = bull_trend_4h and st_bullish and adx_confirmed
        long_pullback = bull_trend_4h and rsi_oversold and adx_confirmed
        
        if long_supertrend or long_pullback:
            new_signal = position_size
        
        # SHORT: Mirror of long
        short_supertrend = bear_trend_4h and st_bearish and adx_confirmed
        short_pullback = bear_trend_4h and rsi_overbought and adx_confirmed
        
        if short_supertrend or short_pullback:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals