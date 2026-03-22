#!/usr/bin/env python3
"""
Experiment #464: 30m Multi-Regime Ensemble with 4h/1d Trend Filter

Hypothesis: 30m timeframe captures intraday swings that 4h/12h miss, but needs
strong HTF filters to avoid whipsaws. This strategy combines:

1. 4H HMA(21) TREND BIAS: Smooth trend filter (HMA less laggy than EMA)
   - Long bias when 30m close > 4h HMA
   - Short bias when 30m close < 4h HMA

2. 1D ADX(14) REGIME DETECTION:
   - ADX > 25 = trending (favor breakout signals)
   - ADX < 20 = ranging (favor mean reversion signals)
   - 20-25 = neutral (require stronger confirmation)

3. THREE 30M SIGNAL TYPES (ensures sufficient trades):
   a) RSI(14) MEAN REVERSION: RSI < 35 (long) or > 65 (short)
      - Looser than 30/70 to ensure trades on BTC/ETH
      - Only in ranging regime (1d ADX < 20)
   
   b) DONCHIAN(20) BREAKOUT: Price breaks 20-bar high/low
      - Only in trending regime (1d ADX > 25)
      - Requires volume confirmation (vol > 1.3x avg)
   
   c) EMA(8/21) CROSSOVER: Fast EMA crosses slow EMA
      - Works in any regime with 4h trend alignment
      - Backup signal when RSI/Donchian don't trigger

4. VOLUME CONFIRMATION:
   - Volume ratio > 1.3 for breakouts (filters false moves)
   - Not required for RSI mean reversion (oversold = opportunity)

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crash protection

6. POSITION SIZING: 0.28 discrete
   - Max 28% capital per position
   - Discrete levels minimize fee churn

Why 30m should work:
- Faster than 4h (captures more swings) but slower than 15m (less noise)
- 4h HMA + 1d ADX provides robust regime detection
- Three signal types ensure >10 trades/symbol/year
- Should work on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_ensemble_4h_hma_1d_adx_rsi_donchian_ema_vol_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio relative to rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg.replace(0, np.inf)
    return vol_ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D ADX REGIME FILTER ===
        adx_1d_val = adx_1d_aligned[i]
        trending_market = adx_1d_val > 25
        ranging_market = adx_1d_val < 20
        # neutral_market = 20 <= adx <= 25
        
        # === SIGNAL 1: RSI MEAN REVERSION (ranging markets) ===
        rsi_long = rsi[i] < 35
        rsi_short = rsi[i] > 65
        
        # === SIGNAL 2: DONCHIAN BREAKOUT (trending markets) ===
        donchian_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_short = close[i] < donchian_lower[i-1] if i > 0 else False
        vol_confirmed = vol_ratio[i] > 1.3
        
        # === SIGNAL 3: EMA CROSSOVER (any regime) ===
        ema_cross_long = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_short = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # === GENERATE SIGNAL (Ensemble - any signal can trigger) ===
        new_signal = 0.0
        
        # RSI MEAN REVERSION (only in ranging market, must align with 4h trend)
        if ranging_market:
            if rsi_long and bull_trend_4h:
                new_signal = SIZE
            elif rsi_short and bear_trend_4h:
                new_signal = -SIZE
        
        # DONCHIAN BREAKOUT (only in trending market with volume confirmation)
        if trending_market and new_signal == 0.0:
            if donchian_long and bull_trend_4h and vol_confirmed:
                new_signal = SIZE
            elif donchian_short and bear_trend_4h and vol_confirmed:
                new_signal = -SIZE
        
        # EMA CROSSOVER (works in any regime, requires 4h trend alignment)
        if new_signal == 0.0:
            if ema_cross_long and bull_trend_4h:
                new_signal = SIZE
            elif ema_cross_short and bear_trend_4h:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
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